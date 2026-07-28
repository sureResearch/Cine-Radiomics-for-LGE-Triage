import os
import shutil
import json
import logging
import random
import platform
import argparse
from functools import partial
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, cohen_kappa_score, log_loss, roc_curve
)
from sklearn.utils import check_random_state
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb
from scipy.stats import uniform, randint

import shap
import scikitplot as skplt
import seaborn as sns
import joblib

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 18,          # 全局字体变大
    "font.weight": "bold",    # 字体加粗
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.unicode_minus": False
})
# -------------------------
# Configuration / Defaults
# -------------------------
SEED = 42
RESULT_DIR = "model_results"
SUBDIRS = ["roc_pr_plots", "shap_plots", "statistical_tests"]
CV_FOLDS = 5
N_ITER_RANDOM_SEARCH = 50
N_JOBS = -1
RANDOM_STATE = SEED
TARGET_NPV = 0.90
TARGET_SENSITIVITY = 0.95
MIN_NEGATIVE_N = 20
MIN_RULEOUT_RATE = 0.05

# -------------------------
# Setup logging & results directories
# -------------------------
os.makedirs(RESULT_DIR, exist_ok=True)
for sub in SUBDIRS:
    os.makedirs(os.path.join(RESULT_DIR, sub), exist_ok=True)

log_path = os.path.join(RESULT_DIR, "training.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("Starting script")

# -------------------------
# Reproducibility & Environment Info
# -------------------------
random.seed(SEED)
np.random.seed(SEED)

env_info = {
    "python_version": platform.python_version(),
    "platform": platform.platform(),
    "numpy_version": np.__version__,
    "pandas_version": pd.__version__,
    "lightgbm_version": lgb.__version__,
    "seed": SEED,
    "script_start": datetime.utcnow().isoformat() + "Z"
}
with open(os.path.join(RESULT_DIR, "env_info.json"), "w", encoding="utf-8") as f:
    json.dump(env_info, f, indent=2)
logger.info(f"Environment info saved")

# -------------------------
# Helper functions
# -------------------------
def check_data_integrity(X, y):
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame")
    s = pd.Series(y)
    if s.isnull().any():
        raise ValueError("Target contains missing values.")
    if X.isnull().any().any():
        logger.warning("Feature matrix contains missing values.")
    unique_vals = sorted(s.unique().tolist())
    if len(unique_vals) != 2:
        raise ValueError("Label is not binary.")

def normalize_binary_labels(y):
    s = pd.to_numeric(pd.Series(y), errors="raise").astype(int)
    if set(s.unique()) != {0, 1}:
        raise ValueError("Label must contain both 0 and 1; event=1, no event=0.")
    return s.values

def safe_div(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else np.nan

def specificity_at_target_sensitivity(estimator, X, y, target_sensitivity):
    y_proba = estimator.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_proba)
    eligible = tpr >= target_sensitivity
    return float(np.max(1 - fpr[eligible])) if np.any(eligible) else 0.0

def select_npv_threshold(
    y_true, y_proba, target_npv=TARGET_NPV,
    target_sensitivity=TARGET_SENSITIVITY,
    min_negative_n=MIN_NEGATIVE_N,
    min_ruleout_rate=MIN_RULEOUT_RATE
):
    rows = []
    for threshold in np.unique(np.r_[0.0, y_proba, 1.0]):
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        negative_n = tn + fn
        ruleout_rate = safe_div(negative_n, len(y_true))
        if negative_n < min_negative_n or ruleout_rate < min_ruleout_rate:
            continue
        rows.append({
            "threshold": float(threshold),
            "sensitivity": safe_div(tp, tp + fn),
            "specificity": safe_div(tn, tn + fp),
            "ppv": safe_div(tp, tp + fp),
            "npv": safe_div(tn, tn + fn),
            "ruleout_rate": ruleout_rate,
            "negative_n": int(negative_n),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
        })

    threshold_df = pd.DataFrame(rows)
    eligible = threshold_df[
        (threshold_df["npv"] >= target_npv) &
        (threshold_df["sensitivity"] >= target_sensitivity)
    ].copy()
    if eligible.empty:
        best = threshold_df.sort_values(
            ["npv", "sensitivity", "ruleout_rate"], ascending=False
        ).head(1)
        detail = best.to_dict("records")[0] if not best.empty else {}
        raise ValueError(
            "No OOF threshold met the prespecified NPV and sensitivity targets. "
            f"Best available candidate: {detail}"
        )
    selected = eligible.sort_values(
        ["ruleout_rate", "specificity", "threshold"],
        ascending=[False, False, False]
    ).iloc[0]
    return float(selected["threshold"]), selected.to_dict()

def compute_binary_metrics(y_true, y_pred_proba, threshold=0.5):
    y_true = np.asarray(y_true)
    y_pred = (y_pred_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "auc": float(roc_auc_score(y_true, y_pred_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "log_loss": float(log_loss(y_true, y_pred_proba)),
        "sensitivity": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "ppv": safe_div(tp, tp + fp),
        "npv": safe_div(tn, tn + fn),
        "ruleout_rate": safe_div(tn + fn, len(y_true)),
        "threshold": float(threshold),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
    }
    return metrics

# -------------------------
# Argument parsing
# -------------------------
parser = argparse.ArgumentParser(description="Reproducible LightGBM training")
parser.add_argument("--train", default=r"C:\Users\cuiro\Desktop\模型构建\Lightgbm\Lightgbm\train.csv")
parser.add_argument("--val", default=r"C:\Users\cuiro\Desktop\模型构建\Lightgbm\Lightgbm\internal.csv")
parser.add_argument("--test", default=r"C:\Users\cuiro\Desktop\模型构建\Lightgbm\Lightgbm\external.csv")
parser.add_argument("--label_col", default="event")
parser.add_argument("--seed", type=int, default=SEED)
args = parser.parse_args()

SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)

# -------------------------
# Load data
# -------------------------
logger.info("Loading data")
train_data = pd.read_csv(args.train)
val_data = pd.read_csv(args.val)
test_data = pd.read_csv(args.test)

for n, df in [("train", train_data), ("val", val_data), ("test", test_data)]:
    if args.label_col not in df.columns:
        raise ValueError(f"Label column missing in {n}")

X_train = train_data.drop(columns=[args.label_col])
y_train = normalize_binary_labels(train_data[args.label_col])

X_val = val_data.drop(columns=[args.label_col])
y_val = normalize_binary_labels(val_data[args.label_col])

X_test = test_data.drop(columns=[args.label_col])
y_test = normalize_binary_labels(test_data[args.label_col])

check_data_integrity(X_train, y_train)
check_data_integrity(X_val, y_val)
check_data_integrity(X_test, y_test)

# -------------------------
# Hyperparameter search space
# -------------------------
param_dist_improved = {
    'n_estimators': randint(300, 900),
    'learning_rate': uniform(0.005, 0.04),
    'max_depth': randint(2, 5),
    'num_leaves': randint(12, 40),
    'min_child_samples': randint(20, 80),
    'min_split_gain': uniform(0.05, 0.25),
    'feature_fraction': uniform(0.6, 0.25),
    'bagging_fraction': uniform(0.6, 0.25),
    'bagging_freq': randint(3, 8),
    'reg_alpha': uniform(0.5, 5.0),
    'reg_lambda': uniform(0.5, 5.0),
    'scale_pos_weight': [1, 2, 3],
    'max_bin': randint(120, 230),
}

lgb_clf = lgb.LGBMClassifier(
    objective='binary',
    metric='auc',
    random_state=SEED,
    n_jobs=N_JOBS,
    verbose=-1
)

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

# -------------------------
# RandomizedSearchCV (NO early stopping here)
# -------------------------
logger.info("Starting RandomizedSearchCV")
ruleout_scorer = partial(
    specificity_at_target_sensitivity,
    target_sensitivity=TARGET_SENSITIVITY
)
random_search = RandomizedSearchCV(
    estimator=lgb_clf,
    param_distributions=param_dist_improved,
    n_iter=N_ITER_RANDOM_SEARCH,
    scoring={
        'roc_auc': 'roc_auc',
        'specificity_at_target_sensitivity': ruleout_scorer
    },
    n_jobs=N_JOBS,
    cv=cv,
    verbose=2,
    random_state=SEED,
    refit='specificity_at_target_sensitivity'
)

random_search.fit(X_train, y_train)
best_params = random_search.best_params_
best_score = random_search.best_score_
logger.info(f"Best params: {best_params}")
logger.info(f"Best CV specificity at sensitivity >= {TARGET_SENSITIVITY:.2f}: {best_score:.4f}")

# Select one locked rule-out threshold from training-set OOF predictions only
oof_model = clone(lgb_clf).set_params(**best_params, n_jobs=1)
oof_proba = cross_val_predict(
    oof_model, X_train, y_train,
    cv=cv, method='predict_proba', n_jobs=N_JOBS
)[:, 1]
optimal_threshold, oof_operating_point = select_npv_threshold(
    y_train, oof_proba
)
logger.info(
    "Locked OOF threshold %.6f | NPV %.4f | sensitivity %.4f | rule-out rate %.4f",
    optimal_threshold,
    oof_operating_point["npv"],
    oof_operating_point["sensitivity"],
    oof_operating_point["ruleout_rate"]
)

# -------------------------
# Final Model Training
# Use exactly the same hyperparameters and number of trees as the OOF models,
# so the locked OOF threshold remains applicable to validation/test predictions.
# The internal and external cohorts are not used during model fitting.
# -------------------------
final_model = clone(lgb_clf).set_params(**best_params)
logger.info("Training final model on the complete training set with locked hyperparameters")
final_model.fit(X_train, y_train)

# -------------------------
# SHAP Analysis
# -------------------------
logger.info("Running SHAP analysis")

X_train_df = X_train.copy()
X_val_df = X_val.copy()
X_test_df = X_test.copy()

explainer = shap.TreeExplainer(final_model)

# 原始 SHAP 输出（可能是 list[2] 或 ndarray）
shap_values_train = explainer.shap_values(X_train_df)
shap_values_val = explainer.shap_values(X_val_df)
shap_values_test = explainer.shap_values(X_test_df)

# 下面这部分仍然提取“阳性类”的 SHAP，用于 Rscore 去除、SHAP CSV 保存等
def _process_shap_values(values):
    if isinstance(values, list) and len(values) == 2:
        return values[0], values[1]  # class 0, class 1
    if isinstance(values, np.ndarray) and len(values.shape) == 2:
        # 单输出时，默认视为 class 1 的 shap，构造一个对称的 class 0
        return -values, values
    raise ValueError("Unexpected SHAP format")

shap_class0_train, shap_class1_train = _process_shap_values(shap_values_train)
shap_class0_val, shap_class1_val = _process_shap_values(shap_values_val)
shap_class0_test, shap_class1_test = _process_shap_values(shap_values_test)

expected_value_raw = explainer.expected_value
if isinstance(expected_value_raw, (list, np.ndarray)):
    expected_array = np.atleast_1d(expected_value_raw)
    expected_value_pos = float(expected_array[1] if expected_array.size > 1 else expected_array[0])
else:
    expected_value_pos = float(expected_value_raw)

# 仍然把“阳性类”shap 保存下来，用于后续去掉 Rscore 计算等
shap_train = shap_class1_train
shap_val = shap_class1_val
shap_test = shap_class1_test

feature_names = list(X_train_df.columns)

def plot_shap_summary(shap_values, X, feature_names, title, save_path):
    """
    绘制 SHAP 全局 summary 图：
    - 对于二分类，传入 list [shap_class0, shap_class1]，shap 会自动做差分并给出全局 summary
    """
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="dot", show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_shap_bar(shap_values, X, feature_names, title, save_path):
    """
    绘制 SHAP 全局 importance bar 图：
    - 对于二分类，传入 list [shap_class0, shap_class1]，shap 会根据全局贡献度给出排序
    """
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# 使用“原始 shap_values_*”（而不是单独的 class1）绘制【全局】summary 和 importance
plot_shap_summary(
    shap_values_train,
    X_train_df,
    feature_names,
    "Training Set - SHAP Global Summary",
    os.path.join(RESULT_DIR, "shap_plots", "train_shap_global_summary.png")
)

plot_shap_bar(
    shap_values_train,
    X_train_df,
    feature_names,
    "Training Set - SHAP Global Feature Importance",
    os.path.join(RESULT_DIR, "shap_plots", "train_shap_global_bar.png")
)

plot_shap_summary(
    shap_values_val,
    X_val_df,
    feature_names,
    "Validation Set - SHAP Global Summary",
    os.path.join(RESULT_DIR, "shap_plots", "val_shap_global_summary.png")
)

plot_shap_bar(
    shap_values_val,
    X_val_df,
    feature_names,
    "Validation Set - SHAP Global Feature Importance",
    os.path.join(RESULT_DIR, "shap_plots", "val_shap_global_bar.png")
)

plot_shap_summary(
    shap_values_test,
    X_test_df,
    feature_names,
    "Test Set - SHAP Global Summary",
    os.path.join(RESULT_DIR, "shap_plots", "test_shap_global_summary.png")
)

plot_shap_bar(
    shap_values_test,
    X_test_df,
    feature_names,
    "Test Set - SHAP Global Feature Importance",
    os.path.join(RESULT_DIR, "shap_plots", "test_shap_global_bar.png")
)

# -------------------------
# Save SHAP arrays (仍然保存 class1 的 SHAP 数组)
# -------------------------
logger.info("Saving SHAP CSV files")

shap_dir = os.path.join(RESULT_DIR, "shap_values")
os.makedirs(shap_dir, exist_ok=True)

def save_shap(shap_array, X_df, name):
    df = pd.DataFrame(shap_array, columns=X_df.columns)
    df.insert(0, "sample_index", X_df.index)
    df.to_csv(os.path.join(shap_dir, f"{name}_shap_values.csv"), index=False)

save_shap(shap_train, X_train_df, "train")
save_shap(shap_val, X_val_df, "val")
save_shap(shap_test, X_test_df, "test")

# -------------------------
# Prediction Results 
# -------------------------
logger.info("Generating prediction results (no test logistic recalibration)")

def generate_prediction_results(model, X, y_true, shap_values=None, expected_value=None, threshold=0.5):
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    results = pd.DataFrame({
        "sample_index": X.index,
        "true_label": y_true,
        "predicted_label": y_pred,
        "predicted_probability": y_proba
    })

    R_FEATURE = "Rscore"
    if shap_values is None or R_FEATURE not in X.columns:
        results["predicted_label_without_Rscore"] = y_pred
        results["predicted_probability_without_Rscore"] = y_proba
        return results

    # 这里仍然用“阳性类”shap_values（shap_train/shap_val/shap_test）来做 Rscore 去除
    r_idx = X.columns.get_loc(R_FEATURE)
    shap_without_r = shap_values.copy()
    shap_without_r[:, r_idx] = 0

    log_odds = expected_value + np.sum(shap_without_r, axis=1)
    prob_without_r = 1 / (1 + np.exp(-log_odds))
    pred_without_r = (prob_without_r >= threshold).astype(int)

    results["predicted_label_without_Rscore"] = pred_without_r
    results["predicted_probability_without_Rscore"] = prob_without_r

    return results

train_results = generate_prediction_results(
    final_model, X_train, y_train, shap_train, expected_value_pos,
    threshold=optimal_threshold
)
val_results = generate_prediction_results(
    final_model, X_val, y_val, shap_val, expected_value_pos,
    threshold=optimal_threshold
)
test_results = generate_prediction_results(
    final_model, X_test, y_test, shap_test, expected_value_pos,
    threshold=optimal_threshold
)

# Save CSV (raw predictions only)
train_results.to_csv(os.path.join(RESULT_DIR, "prediction_results_train.csv"), index=False)
val_results.to_csv(os.path.join(RESULT_DIR, "prediction_results_validation.csv"), index=False)
test_results.to_csv(os.path.join(RESULT_DIR, "prediction_results_test.csv"), index=False)

logger.info("Prediction results saved")

# -------------------------
# Evaluate (No bootstrap)
# -------------------------
def evaluate_basic(model, X, y, name, threshold):
    y_proba = model.predict_proba(X)[:, 1]
    return compute_binary_metrics(y, y_proba, threshold=threshold)

train_eval = evaluate_basic(final_model, X_train, y_train, "train", optimal_threshold)
val_eval = evaluate_basic(final_model, X_val, y_val, "val", optimal_threshold)
test_eval = evaluate_basic(final_model, X_test, y_test, "test", optimal_threshold)

results_summary = {
    "model_info": {
        "best_params": best_params,
        "best_cv_score": float(best_score),
        "best_cv_score_name": "specificity_at_target_sensitivity",
        "target_npv": TARGET_NPV,
        "target_sensitivity": TARGET_SENSITIVITY,
        "selected_threshold": optimal_threshold,
        "oof_operating_point": oof_operating_point
    },
    "train": train_eval,
    "validation": val_eval,
    "test": test_eval
}

with open(os.path.join(RESULT_DIR, "statistical_tests", "evaluation_summary.json"), "w", encoding="utf-8") as f:
    json.dump(results_summary, f, indent=2)

logger.info("Evaluation summary saved")

# -------------------------
# Save Model
# -------------------------
final_model.selected_threshold_ = optimal_threshold
joblib.dump(final_model, os.path.join(RESULT_DIR, "best_model.joblib"))
logger.info("Model saved")

# -------------------------
# ROC & PR curves
# -------------------------
def plot_roc_pr_scikit(y_true, y_proba, name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    proba_mat = np.vstack([1 - y_proba, y_proba]).T

    # ROC curve
    skplt.metrics.plot_roc(y_true, proba_mat)
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    plt.title(f"{name} ROC Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name.lower().replace(' ', '_')}_roc_curve.png"), dpi=400)
    plt.close()

    # PR curve
    skplt.metrics.plot_precision_recall(y_true, proba_mat)
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    plt.title(f"{name} PR Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name.lower().replace(' ', '_')}_pr_curve.png"), dpi=400)
    plt.close()

roc_dir = os.path.join(RESULT_DIR, "roc_pr_plots")
plot_roc_pr_scikit(y_train, final_model.predict_proba(X_train)[:, 1], "Training Set", roc_dir)
plot_roc_pr_scikit(y_val, final_model.predict_proba(X_val)[:, 1], "Validation Set", roc_dir)
plot_roc_pr_scikit(y_test, final_model.predict_proba(X_test)[:, 1], "Test Set", roc_dir)

logger.info("ROC & PR saved")
logger.info("Script finished successfully")
