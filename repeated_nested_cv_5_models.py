# -*- coding: utf-8 -*-
"""
Five-model repeated nested cross-validation for LGE rule-out model selection.

Models:
1. LightGBM
2. SVM
3. XGBoost
4. CatBoost
5. Bagging

Design:
- Outer loop: 5-fold stratified CV repeated 5 times.
- Inner loop: 5-fold stratified randomized hyperparameter search.
- Hyperparameters are selected by specificity at sensitivity >= 0.95,
  with AUC and Brier score used as tie-breakers.
- Within each outer-training fold, the operating threshold is derived from
  inner out-of-fold predictions by maximizing specificity at sensitivity >= 0.95.
- The locked fold-specific threshold is applied to the untouched outer-validation fold.
- Internal and external validation datasets are not used.
"""

import inspect
import json
import os
import random
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import lightgbm as lgb
import xgboost as xgb
import catboost as cb


# =========================
# 1. Configuration
# =========================
DATA_DIR = Path(r"C:\Users\cuiro\Desktop\LGE01数据分析\模型构建\other models\重复嵌套交叉")
TRAIN_FILE = DATA_DIR / "train.csv"
RESULT_DIR = DATA_DIR / "repeated_nested_cv_results"
CHECKPOINT_DIR = RESULT_DIR / "checkpoints"

LABEL_COL = "event"
TARGET_SENSITIVITY = 0.95

OUTER_FOLDS = 5
OUTER_REPEATS = 5
INNER_FOLDS = 5
N_ITER_SEARCH = 30
N_BOOTSTRAPS = 2000

SEED = 42
N_JOBS = -1
RESUME = True

MODEL_ORDER = ["LightGBM", "SVM", "XGBoost", "CatBoost", "Bagging"]
TABLE_METRICS = [
    "AUC", "Accuracy", "Sensitivity", "Specificity",
    "PPV", "NPV", "F1 Score", "Kappa"
]
ALL_METRICS = TABLE_METRICS + [
    "Brier Score", "Specificity@Sensitivity>=0.95"
]


# =========================
# 2. Utility functions
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def safe_div(numerator, denominator):
    return float(numerator / denominator) if denominator else np.nan


def validate_data(df):
    if LABEL_COL not in df.columns:
        raise ValueError(f"Missing outcome column: {LABEL_COL}")

    y = pd.to_numeric(df[LABEL_COL], errors="raise").astype(int)
    if set(y.unique()) != {0, 1}:
        raise ValueError(f"{LABEL_COL} must contain both 0 and 1.")

    X = df.drop(columns=[LABEL_COL]).copy()
    if X.empty:
        raise ValueError("No predictor columns were found.")

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="raise")

    if X.isna().any().any():
        missing = X.isna().sum()
        missing = missing[missing > 0].to_dict()
        raise ValueError(f"Predictors contain missing values: {missing}")

    return X, y.to_numpy()


def specificity_at_target_sensitivity_from_prob(y_true, y_prob, target=0.95):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    eligible = tpr >= target
    if not np.any(eligible):
        return 0.0
    return float(np.max(1.0 - fpr[eligible]))


def specificity_at_target_sensitivity_scorer(estimator, X, y):
    y_prob = estimator.predict_proba(X)[:, 1]
    return specificity_at_target_sensitivity_from_prob(
        y, y_prob, TARGET_SENSITIVITY
    )


def select_refit_index(cv_results):
    """Primary: specificity at sensitivity >=0.95; ties: AUC, then Brier."""
    scores = pd.DataFrame({
        "index": np.arange(len(cv_results["params"])),
        "spec95": np.asarray(cv_results["mean_test_spec95"], dtype=float),
        "auc": np.asarray(cv_results["mean_test_auc"], dtype=float),
        "neg_brier": np.asarray(cv_results["mean_test_neg_brier"], dtype=float),
    }).replace([np.inf, -np.inf], np.nan).fillna(-np.inf)

    scores = scores.sort_values(
        ["spec95", "auc", "neg_brier"],
        ascending=[False, False, False]
    )
    return int(scores.iloc[0]["index"])


def select_threshold(y_true, y_prob, target=0.95):
    """
    Choose the highest-specificity threshold among thresholds with sensitivity >= target.
    Ties are resolved by choosing the highest threshold.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    eligible = np.where(tpr >= target)[0]
    if len(eligible) == 0:
        raise RuntimeError("No threshold achieved the target sensitivity.")

    specificity = 1.0 - fpr
    best_specificity = np.max(specificity[eligible])
    candidates = eligible[np.isclose(
        specificity[eligible], best_specificity, rtol=0, atol=1e-12
    )]

    finite_candidates = candidates[np.isfinite(thresholds[candidates])]
    if len(finite_candidates) > 0:
        best_index = finite_candidates[np.argmax(thresholds[finite_candidates])]
    else:
        best_index = candidates[0]

    threshold = float(np.clip(thresholds[best_index], 0.0, 1.0))
    return threshold


def metrics_from_prob_and_pred(y_true, y_prob, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = np.asarray(y_pred, dtype=int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "AUC": float(roc_auc_score(y_true, y_prob)),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Sensitivity": safe_div(tp, tp + fn),
        "Specificity": safe_div(tn, tn + fp),
        "PPV": safe_div(tp, tp + fp),
        "NPV": safe_div(tn, tn + fn),
        "F1 Score": float(f1_score(y_true, y_pred, zero_division=0)),
        "Kappa": float(cohen_kappa_score(y_true, y_pred)),
        "Brier Score": float(brier_score_loss(y_true, y_prob)),
        "Specificity@Sensitivity>=0.95":
            specificity_at_target_sensitivity_from_prob(
                y_true, y_prob, TARGET_SENSITIVITY
            ),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }




class CompatibleCatBoostClassifier(ClassifierMixin, BaseEstimator):
    """CatBoost wrapper compatible with recent and older scikit-learn versions."""

    def __init__(
        self,
        iterations=500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        random_strength=1.0,
        border_count=128,
        subsample=0.8,
        scale_pos_weight=1.0,
        random_seed=42,
        thread_count=1,
        verbose=False,
        allow_writing_files=False,
    ):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.random_strength = random_strength
        self.border_count = border_count
        self.subsample = subsample
        self.scale_pos_weight = scale_pos_weight
        self.random_seed = random_seed
        self.thread_count = thread_count
        self.verbose = verbose
        self.allow_writing_files = allow_writing_files

    def fit(self, X, y):
        self.model_ = cb.CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            bootstrap_type="Bernoulli",
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            random_strength=self.random_strength,
            border_count=self.border_count,
            subsample=self.subsample,
            scale_pos_weight=self.scale_pos_weight,
            random_seed=self.random_seed,
            thread_count=self.thread_count,
            verbose=self.verbose,
            allow_writing_files=self.allow_writing_files,
        )
        self.model_.fit(X, y)
        self.classes_ = np.asarray(self.model_.classes_)
        self.n_features_in_ = X.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.model_.predict(X).astype(int).ravel()


# =========================
# 3. Candidate models
# =========================
def make_bagging(seed):
    tree = DecisionTreeClassifier(random_state=seed)
    signature = inspect.signature(BaggingClassifier)
    if "estimator" in signature.parameters:
        model = BaggingClassifier(
            estimator=tree, random_state=seed, n_jobs=1
        )
        prefix = "estimator__"
    else:
        model = BaggingClassifier(
            base_estimator=tree, random_state=seed, n_jobs=1
        )
        prefix = "base_estimator__"

    params = {
        "n_estimators": randint(80, 401),
        "max_samples": uniform(0.60, 0.40),
        "max_features": uniform(0.60, 0.40),
        "bootstrap": [True],
        "bootstrap_features": [False, True],
        f"{prefix}max_depth": randint(2, 9),
        f"{prefix}min_samples_split": randint(2, 21),
        f"{prefix}min_samples_leaf": randint(1, 16),
        f"{prefix}class_weight": [None, "balanced"],
    }
    return model, params


def make_model_and_space(model_name, seed):
    if model_name == "LightGBM":
        model = lgb.LGBMClassifier(
            objective="binary",
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
        )
        params = {
            "n_estimators": randint(250, 901),
            "learning_rate": loguniform(0.005, 0.08),
            "max_depth": randint(2, 6),
            "num_leaves": randint(6, 33),
            "min_child_samples": randint(10, 81),
            "subsample": uniform(0.60, 0.40),
            "colsample_bytree": uniform(0.60, 0.40),
            "reg_alpha": loguniform(1e-3, 10),
            "reg_lambda": loguniform(1e-3, 10),
            "scale_pos_weight": [1, 2, 3],
        }
        return model, params

    if model_name == "SVM":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(
                kernel="rbf",
                probability=True,
                random_state=seed
            )),
        ])
        params = {
            "svc__C": loguniform(1e-2, 1e3),
            "svc__gamma": loguniform(1e-4, 1),
            "svc__class_weight": [None, "balanced"],
        }
        return model, params

    if model_name == "XGBoost":
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=seed,
            n_jobs=1,
        )
        params = {
            "n_estimators": randint(250, 901),
            "learning_rate": loguniform(0.005, 0.08),
            "max_depth": randint(2, 7),
            "min_child_weight": loguniform(0.5, 10),
            "gamma": uniform(0, 5),
            "subsample": uniform(0.60, 0.40),
            "colsample_bytree": uniform(0.60, 0.40),
            "reg_alpha": loguniform(1e-3, 10),
            "reg_lambda": loguniform(1e-2, 10),
            "scale_pos_weight": [1, 2, 3],
        }
        return model, params

    if model_name == "CatBoost":
        model = CompatibleCatBoostClassifier(
            random_seed=seed,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
        )
        params = {
            "iterations": randint(250, 901),
            "learning_rate": loguniform(0.005, 0.08),
            "depth": randint(3, 8),
            "l2_leaf_reg": loguniform(0.5, 20),
            "random_strength": uniform(0, 3),
            "border_count": [64, 128, 254],
            "subsample": uniform(0.60, 0.40),
            "scale_pos_weight": [1, 2, 3],
        }
        return model, params

    if model_name == "Bagging":
        return make_bagging(seed)

    raise ValueError(f"Unknown model: {model_name}")


# =========================
# 4. One nested-CV outer fold
# =========================
def run_outer_fold(
    model_name, X, y, train_idx, valid_idx,
    repeat_id, fold_id
):
    checkpoint = CHECKPOINT_DIR / (
        f"{model_name}_repeat_{repeat_id:02d}_fold_{fold_id:02d}.joblib"
    )

    if RESUME and checkpoint.exists():
        return joblib.load(checkpoint)

    fold_seed = SEED + repeat_id * 1000 + fold_id * 100 + MODEL_ORDER.index(model_name)
    set_seed(fold_seed)

    X_outer_train = X.iloc[train_idx]
    y_outer_train = y[train_idx]
    X_outer_valid = X.iloc[valid_idx]
    y_outer_valid = y[valid_idx]

    estimator, param_space = make_model_and_space(model_name, fold_seed)
    inner_cv = StratifiedKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=fold_seed
    )

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_space,
        n_iter=N_ITER_SEARCH,
        scoring={
            "spec95": specificity_at_target_sensitivity_scorer,
            "auc": "roc_auc",
            "neg_brier": "neg_brier_score",
        },
        refit=select_refit_index,
        cv=inner_cv,
        random_state=fold_seed,
        n_jobs=N_JOBS,
        verbose=0,
        error_score="raise",
        return_train_score=False,
    )
    search.fit(X_outer_train, y_outer_train)

    tuned_model = clone(search.best_estimator_)

    inner_oof_prob = cross_val_predict(
        tuned_model,
        X_outer_train,
        y_outer_train,
        cv=inner_cv,
        method="predict_proba",
        n_jobs=N_JOBS,
    )[:, 1]

    threshold = select_threshold(
        y_outer_train,
        inner_oof_prob,
        TARGET_SENSITIVITY
    )

    tuned_model.fit(X_outer_train, y_outer_train)
    valid_prob = tuned_model.predict_proba(X_outer_valid)[:, 1]
    valid_pred = (valid_prob >= threshold).astype(int)

    fold_metrics = metrics_from_prob_and_pred(
        y_outer_valid, valid_prob, valid_pred
    )

    result = {
        "model": model_name,
        "repeat": repeat_id,
        "fold": fold_id,
        "train_n": int(len(train_idx)),
        "validation_n": int(len(valid_idx)),
        "threshold": threshold,
        "validation_indices": np.asarray(valid_idx, dtype=int),
        "validation_probabilities": np.asarray(valid_prob, dtype=float),
        "validation_predictions": np.asarray(valid_pred, dtype=int),
        "best_params": json_safe(search.best_params_),
        "inner_best_spec95": float(
            search.cv_results_["mean_test_spec95"][search.best_index_]
        ),
        "inner_best_auc": float(
            search.cv_results_["mean_test_auc"][search.best_index_]
        ),
        "inner_best_brier": float(
            -search.cv_results_["mean_test_neg_brier"][search.best_index_]
        ),
        "metrics": fold_metrics,
    }

    joblib.dump(result, checkpoint)
    return result


# =========================
# 5. Repeated nested CV
# =========================
def run_repeated_nested_cv(X, y):
    n = len(y)
    probability_store = {
        model: np.full((OUTER_REPEATS, n), np.nan)
        for model in MODEL_ORDER
    }
    prediction_store = {
        model: np.full((OUTER_REPEATS, n), np.nan)
        for model in MODEL_ORDER
    }

    fold_rows = []
    parameter_rows = []

    for repeat_id in range(1, OUTER_REPEATS + 1):
        outer_cv = StratifiedKFold(
            n_splits=OUTER_FOLDS,
            shuffle=True,
            random_state=SEED + repeat_id
        )

        for fold_id, (train_idx, valid_idx) in enumerate(
            outer_cv.split(X, y), start=1
        ):
            print(
                f"\nRepeat {repeat_id}/{OUTER_REPEATS}, "
                f"outer fold {fold_id}/{OUTER_FOLDS}"
            )

            for model_name in MODEL_ORDER:
                print(f"  Running {model_name}...")
                result = run_outer_fold(
                    model_name=model_name,
                    X=X,
                    y=y,
                    train_idx=train_idx,
                    valid_idx=valid_idx,
                    repeat_id=repeat_id,
                    fold_id=fold_id,
                )

                idx = result["validation_indices"]
                probability_store[model_name][repeat_id - 1, idx] = (
                    result["validation_probabilities"]
                )
                prediction_store[model_name][repeat_id - 1, idx] = (
                    result["validation_predictions"]
                )

                row = {
                    "Model": model_name,
                    "Repeat": repeat_id,
                    "Outer fold": fold_id,
                    "Threshold": result["threshold"],
                    "Inner CV specificity@95% sensitivity":
                        result["inner_best_spec95"],
                    "Inner CV AUC": result["inner_best_auc"],
                    "Inner CV Brier score": result["inner_best_brier"],
                }
                row.update({
                    key: value for key, value in result["metrics"].items()
                    if key not in {"TP", "FP", "TN", "FN"}
                })
                row.update({
                    key: result["metrics"][key]
                    for key in ["TP", "FP", "TN", "FN"]
                })
                fold_rows.append(row)

                parameter_rows.append({
                    "Model": model_name,
                    "Repeat": repeat_id,
                    "Outer fold": fold_id,
                    "Best parameters": json.dumps(
                        result["best_params"],
                        ensure_ascii=False,
                        sort_keys=True
                    )
                })

    for model_name in MODEL_ORDER:
        if np.isnan(probability_store[model_name]).any():
            raise RuntimeError(f"Incomplete OOF probabilities for {model_name}.")
        if np.isnan(prediction_store[model_name]).any():
            raise RuntimeError(f"Incomplete OOF predictions for {model_name}.")

    return (
        probability_store,
        prediction_store,
        pd.DataFrame(fold_rows),
        pd.DataFrame(parameter_rows),
    )


# =========================
# 6. Repeated-CV summaries and bootstrap CI
# =========================
def calculate_repeat_metrics(y, probability_store, prediction_store):
    rows = []

    for model_name in MODEL_ORDER:
        for repeat_id in range(OUTER_REPEATS):
            metrics = metrics_from_prob_and_pred(
                y,
                probability_store[model_name][repeat_id],
                prediction_store[model_name][repeat_id].astype(int),
            )
            row = {
                "Model": model_name,
                "Repeat": repeat_id + 1,
            }
            row.update(metrics)
            rows.append(row)

    return pd.DataFrame(rows)


def bootstrap_summary(y, probability_store, prediction_store):
    rng = np.random.RandomState(SEED + 9999)
    class0 = np.where(y == 0)[0]
    class1 = np.where(y == 1)[0]
    summary_rows = []

    for model_name in MODEL_ORDER:
        repeat_point_metrics = []
        for repeat_id in range(OUTER_REPEATS):
            repeat_point_metrics.append(
                metrics_from_prob_and_pred(
                    y,
                    probability_store[model_name][repeat_id],
                    prediction_store[model_name][repeat_id].astype(int),
                )
            )

        point_estimates = {
            metric: float(np.mean([
                repeat_result[metric]
                for repeat_result in repeat_point_metrics
            ]))
            for metric in ALL_METRICS
        }

        bootstrap_values = {
            metric: [] for metric in ALL_METRICS
        }

        for _ in range(N_BOOTSTRAPS):
            sampled_indices = np.concatenate([
                rng.choice(class0, size=len(class0), replace=True),
                rng.choice(class1, size=len(class1), replace=True),
            ])

            bootstrap_repeat_metrics = []
            for repeat_id in range(OUTER_REPEATS):
                bootstrap_repeat_metrics.append(
                    metrics_from_prob_and_pred(
                        y[sampled_indices],
                        probability_store[model_name][repeat_id, sampled_indices],
                        prediction_store[model_name][
                            repeat_id, sampled_indices
                        ].astype(int),
                    )
                )

            for metric in ALL_METRICS:
                bootstrap_values[metric].append(
                    np.mean([
                        repeat_result[metric]
                        for repeat_result in bootstrap_repeat_metrics
                    ])
                )

        for metric in ALL_METRICS:
            lower, upper = np.percentile(
                bootstrap_values[metric], [2.5, 97.5]
            )
            estimate = point_estimates[metric]
            summary_rows.append({
                "Model": model_name,
                "Metric": metric,
                "Estimate": estimate,
                "CI lower": float(lower),
                "CI upper": float(upper),
                "Estimate [95% CI]":
                    f"{estimate:.3f} [{lower:.3f}–{upper:.3f}]",
            })

    return pd.DataFrame(summary_rows)


def make_ranking(summary):
    wide = summary.pivot(
        index="Model", columns="Metric", values="Estimate"
    ).reset_index()

    ranking = wide.sort_values(
        [
            "Specificity@Sensitivity>=0.95",
            "AUC",
            "Brier Score",
        ],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    ranking.insert(0, "Rank", np.arange(1, len(ranking) + 1))
    ranking["Selection rule"] = (
        "Highest nested-CV specificity at sensitivity >=0.95; "
        "ties resolved by higher AUC and lower Brier score"
    )
    return ranking


def make_supplement_table(summary):
    lookup = summary.pivot(
        index="Metric",
        columns="Model",
        values="Estimate [95% CI]"
    )

    table = pd.DataFrame({"Metric": TABLE_METRICS})
    for model_name in MODEL_ORDER:
        table[model_name] = [
            lookup.loc[metric, model_name]
            for metric in TABLE_METRICS
        ]
    return table


def make_prediction_table(y, probability_store, prediction_store):
    rows = []
    for model_name in MODEL_ORDER:
        for repeat_id in range(OUTER_REPEATS):
            temp = pd.DataFrame({
                "Model": model_name,
                "Repeat": repeat_id + 1,
                "Row index": np.arange(len(y)),
                "True label": y,
                "OOF probability":
                    probability_store[model_name][repeat_id],
                "OOF predicted label":
                    prediction_store[model_name][repeat_id].astype(int),
            })
            rows.append(temp)
    return pd.concat(rows, ignore_index=True)


# =========================
# 7. Save final outputs
# =========================
def save_outputs(
    summary, ranking, supplement_table,
    repeat_metrics, fold_metrics,
    parameter_table, prediction_table
):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    output_xlsx = RESULT_DIR / "Repeated_nested_CV_results.xlsx"
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        supplement_table.to_excel(
            writer, sheet_name="Supplemental Table S6", index=False
        )
        ranking.to_excel(
            writer, sheet_name="Model ranking", index=False
        )
        summary.to_excel(
            writer, sheet_name="Summary long", index=False
        )
        repeat_metrics.to_excel(
            writer, sheet_name="Repeat metrics", index=False
        )
        fold_metrics.to_excel(
            writer, sheet_name="Outer fold metrics", index=False
        )
        parameter_table.to_excel(
            writer, sheet_name="Best parameters", index=False
        )

        for sheet_name, worksheet in writer.sheets.items():
            worksheet.freeze_panes = "A2"
            for column_cells in worksheet.columns:
                max_length = min(
                    max(
                        len(str(cell.value)) if cell.value is not None else 0
                        for cell in column_cells
                    ) + 2,
                    60
                )
                worksheet.column_dimensions[
                    column_cells[0].column_letter
                ].width = max_length

    prediction_table.to_csv(
        RESULT_DIR / "Repeated_nested_CV_OOF_predictions.csv",
        index=False
    )

    summary.to_csv(
        RESULT_DIR / "Repeated_nested_CV_summary.csv",
        index=False
    )

    config = {
        "train_file": str(TRAIN_FILE),
        "label_column": LABEL_COL,
        "models": MODEL_ORDER,
        "target_sensitivity": TARGET_SENSITIVITY,
        "outer_folds": OUTER_FOLDS,
        "outer_repeats": OUTER_REPEATS,
        "inner_folds": INNER_FOLDS,
        "random_search_iterations": N_ITER_SEARCH,
        "bootstrap_resamples": N_BOOTSTRAPS,
        "seed": SEED,
        "selection_rule":
            "Highest specificity at sensitivity >=0.95; "
            "ties by AUC and Brier score",
    }
    with open(
        RESULT_DIR / "run_config.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(config, file, ensure_ascii=False, indent=2)

    print("\nCompleted.")
    print(f"Main results: {output_xlsx}")
    print(f"OOF predictions: {RESULT_DIR / 'Repeated_nested_CV_OOF_predictions.csv'}")
    print("\nModel ranking:")
    print(ranking[[
        "Rank", "Model",
        "Specificity@Sensitivity>=0.95",
        "AUC", "Brier Score"
    ]].to_string(index=False))


# =========================
# 8. Main
# =========================
def main():
    warnings.filterwarnings("ignore")
    set_seed(SEED)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")

    data = pd.read_csv(TRAIN_FILE)
    X, y = validate_data(data)

    print(f"Training file: {TRAIN_FILE}")
    print(f"Sample size: {len(y)}")
    print(f"Predictors ({X.shape[1]}): {', '.join(X.columns)}")
    print(f"LGE-positive: {int(np.sum(y == 1))}")
    print(f"LGE-negative: {int(np.sum(y == 0))}")
    print(
        f"Design: {OUTER_REPEATS} repeats × {OUTER_FOLDS}-fold outer CV; "
        f"{INNER_FOLDS}-fold inner CV"
    )

    probability_store, prediction_store, fold_metrics, parameter_table = (
        run_repeated_nested_cv(X, y)
    )

    repeat_metrics = calculate_repeat_metrics(
        y, probability_store, prediction_store
    )
    summary = bootstrap_summary(
        y, probability_store, prediction_store
    )
    ranking = make_ranking(summary)
    supplement_table = make_supplement_table(summary)
    prediction_table = make_prediction_table(
        y, probability_store, prediction_store
    )

    save_outputs(
        summary=summary,
        ranking=ranking,
        supplement_table=supplement_table,
        repeat_metrics=repeat_metrics,
        fold_metrics=fold_metrics,
        parameter_table=parameter_table,
        prediction_table=prediction_table,
    )


if __name__ == "__main__":
    main()
