import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, confusion_matrix,
    precision_score, recall_score, f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
import lightgbm as lgb
from xgboost import XGBClassifier
import shap
import lime
from tqdm import tqdm


import numpy as np
import warnings
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score

def compute_metrics(y_true, y_prob, threshold=0.5, n_bootstrap=1000, random_state=42):
    """
    Compute:
      - AUC and 95% CI (bootstrap)
      - Precision, Recall, F1 (no CI)
      - Confusion matrix (based on threshold)
    NOTE: y_true and y_prob will be coerced to numpy arrays to avoid pandas label-indexing issues.
    """
    # ---- 强制为 numpy array，按位置索引 ----
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length.")

    # 二分类预测（按阈值）
    y_pred = (y_prob >= threshold).astype(int)

    # ---- AUC（可能因为单一类别而无法计算） ----
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception as e:
        warnings.warn(f"AUC could not be computed: {e}")
        auc = float("nan")

    # ---- bootstrap CI for AUC ----
    bootstrapped_scores = []
    rng = np.random.default_rng(random_state)

    # 如果 auc 无法计算（例如 y_true 全为同一类），跳过 bootstrap
    if not np.isnan(auc):
        for _ in range(n_bootstrap):
            # 有放回抽样位置索引（positional）
            idx = rng.choice(len(y_true), size=len(y_true), replace=True)
            # 如果抽样样本只包含单一类，跳过这次抽样
            if len(np.unique(y_true[idx])) < 2:
                continue
            try:
                s = roc_auc_score(y_true[idx], y_prob[idx])
                bootstrapped_scores.append(s)
            except Exception:
                # 某些罕见情况（如常数预测）会抛异常，直接跳过
                continue

    if len(bootstrapped_scores) >= 2:
        ci_lower, ci_upper = np.percentile(bootstrapped_scores, [2.5, 97.5])
    else:
        ci_lower, ci_upper = (float("nan"), float("nan"))

    # ---- 其它分类指标 ----
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "AUC": auc,
        "AUC_CI95": (ci_lower, ci_upper),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ConfusionMatrix": cm,
        "n_bootstrap_used": len(bootstrapped_scores)
    }



class ModelTrainer:
    def __init__(self, X_train, Y_train, X_test, Y_test):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

    def train_logistic_regression(self):
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(self.X_train, self.Y_train)

        prob_train = model.predict_proba(self.X_train)[:, 1]
        prob_test = model.predict_proba(self.X_test)[:, 1]

        metrics_train = compute_metrics(self.Y_train, prob_train)
        metrics_test = compute_metrics(self.Y_test, prob_test)

        return model, metrics_train, metrics_test

    def train_xgboost(self):
        model = XGBClassifier(
            objective="binary:logistic",
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.05,
            reg_lambda=0.1,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42
        )
        model.fit(self.X_train, self.Y_train)

        prob_train = model.predict_proba(self.X_train)[:, 1]
        prob_test = model.predict_proba(self.X_test)[:, 1]

        metrics_train = compute_metrics(self.Y_train, prob_train)
        metrics_test = compute_metrics(self.Y_test, prob_test)

        return model, metrics_train, metrics_test

    def train_random_forest(self):
        model = RandomForestClassifier(
            n_estimators=500, max_depth=7, min_samples_split=20, random_state=42
        )
        model.fit(self.X_train, self.Y_train)

        prob_train = model.predict_proba(self.X_train)[:, 1]
        prob_test = model.predict_proba(self.X_test)[:, 1]

        metrics_train = compute_metrics(self.Y_train, prob_train)
        metrics_test = compute_metrics(self.Y_test, prob_test)

        return model, metrics_train, metrics_test

    def train_ada_boost(self):
        model = AdaBoostClassifier(
            n_estimators=200, learning_rate=0.5, random_state=42
        )
        model.fit(self.X_train, self.Y_train)

        prob_train = model.predict_proba(self.X_train)[:, 1]
        prob_test = model.predict_proba(self.X_test)[:, 1]

        metrics_train = compute_metrics(self.Y_train, prob_train)
        metrics_test = compute_metrics(self.Y_test, prob_test)

        return model, metrics_train, metrics_test

    def train_naive_bayes(self):
        model = GaussianNB()
        model.fit(self.X_train, self.Y_train)

        prob_train = model.predict_proba(self.X_train)[:, 1]
        prob_test = model.predict_proba(self.X_test)[:, 1]

        metrics_train = compute_metrics(self.Y_train, prob_train)
        metrics_test = compute_metrics(self.Y_test, prob_test)

        return model, metrics_train, metrics_test

    def train_svm(self):
        model = SVC(kernel="rbf", C=1, probability=True, random_state=42)
        model.fit(self.X_train, self.Y_train)

        prob_train = model.predict_proba(self.X_train)[:, 1]
        prob_test = model.predict_proba(self.X_test)[:, 1]

        metrics_train = compute_metrics(self.Y_train, prob_train)
        metrics_test = compute_metrics(self.Y_test, prob_test)

        return model, metrics_train, metrics_test

    def train_lightgbm(self):
        model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            colsample_bytree=0.8,
            subsample=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42
        )
        model.fit(self.X_train, self.Y_train)

        prob_train = model.predict_proba(self.X_train)[:, 1]
        prob_test = model.predict_proba(self.X_test)[:, 1]

        metrics_train = compute_metrics(self.Y_train, prob_train)
        metrics_test = compute_metrics(self.Y_test, prob_test)

        return model, metrics_train, metrics_test
