import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
import lightgbm as lgb
from xgboost import XGBClassifier
import warnings
from imblearn.over_sampling import SMOTE
from sklearn.metrics import brier_score_loss
from content.model_explainers import ModelExplainer


class ModelTrainer:
    def __init__(self, X_train, Y_train, X_test, Y_test, use_smote=False, smote_k_neighbors=5, random_state=42):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.final_selected_features = [
            "sofa", "apsiii", "fluid_in_48h_ml", "uo_mlkghr",
            "chemistry_creatinine", "chemistry_sodium", "chemistry_aniongap",
            "rdw", "pt", "ph", "spo2", "resp_rate"
        ]
        if use_smote:
            smote = SMOTE(k_neighbors=smote_k_neighbors, random_state=random_state)
            X_res, y_res = smote.fit_resample(X_train, Y_train)
            self.X_train = X_res
            self.Y_train = y_res
        else:
            self.X_train = X_train
            self.Y_train = Y_train

        print(self.X_train.shape)

    def explainability(self, model, task='Mortality'):
        explainer = ModelExplainer(
            model=model,
            X_train=self.X_train,
            X_test=self.X_test,
            Y_test=self.Y_test,
            feature_names=self.final_selected_features,
            task=task
        )
        explainer.explain_shap(plot_summary=True)
        explainer.explain_lime(instance_idx=0)
        ablation_results = explainer.ablation_study(model, plot=True, top_n=12)
        print(ablation_results)

    def compute_metrics(self, y_true, y_prob, threshold=0.5, n_bootstrap=1000, random_state=42):
        y_true = np.asarray(y_true).ravel()
        y_prob = np.asarray(y_prob).ravel()
        if len(y_true) != len(y_prob):
            raise ValueError("y_true and y_prob must have the same length.")
        y_pred = (y_prob >= threshold).astype(int)
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception as e:
            warnings.warn(f"AUC could not be computed: {e}")
            auc = float("nan")
        bootstrapped_scores = []
        rng = np.random.default_rng(random_state)
        if not np.isnan(auc):
            for _ in range(n_bootstrap):
                idx = rng.choice(len(y_true), size=len(y_true), replace=True)
                if len(np.unique(y_true[idx])) < 2:
                    continue
                try:
                    s = roc_auc_score(y_true[idx], y_prob[idx])
                    bootstrapped_scores.append(s)
                except Exception:
                    continue
        if len(bootstrapped_scores) >= 2:
            ci_lower, ci_upper = np.percentile(bootstrapped_scores, [2.5, 97.5])
        else:
            ci_lower, ci_upper = (float("nan"), float("nan"))
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        brier = brier_score_loss(y_true, y_prob)
        return {
            "AUC": auc,
            "AUC_CI95": (ci_lower, ci_upper),
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Brier": brier,
            "ConfusionMatrix": cm,
            "n_bootstrap_used": len(bootstrapped_scores)
        }
    
    def train_logistic_regression(self):
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(self.X_train, self.Y_train)
        prob_train = model.predict_proba(self.X_train)[:, 1]
        prob_test = model.predict_proba(self.X_test)[:, 1]
        metrics_train = self.compute_metrics(self.Y_train, prob_train)
        metrics_test = self.compute_metrics(self.Y_test, prob_test)
        return model, metrics_train, metrics_test

    def train_xgboost(self):
        model = XGBClassifier(
            objective="binary:logistic",
            n_estimators=5000,  # large cap; early stopping will decide actual trees
            learning_rate=0.02,
            max_depth=4,  # lower depth to reduce overfitting
            min_child_weight=5,  # stronger regularization
            gamma=0.5,  # require gain to split
            subsample=0.75,
            colsample_bytree=0.75,
            reg_alpha=0.2,
            reg_lambda=2.0,
            eval_metric="auc",
            random_state=42,
            n_jobs=-1
        )
        model.fit(self.X_train, self.Y_train)
        prob_train = model.predict_proba(self.X_train)[:, 1]
        prob_test = model.predict_proba(self.X_test)[:, 1]
        metrics_train = self.compute_metrics(self.Y_train, prob_train)
        metrics_test = self.compute_metrics(self.Y_test, prob_test)
        return model, metrics_train, metrics_test

    def train_random_forest(self):
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=7,
            min_samples_split=20,
            random_state=42
        )
        model.fit(self.X_train, self.Y_train)
        prob_train = model.predict_proba(self.X_train)[:, 1]
        prob_test = model.predict_proba(self.X_test)[:, 1]
        metrics_train = self.compute_metrics(self.Y_train, prob_train)
        metrics_test = self.compute_metrics(self.Y_test, prob_test)
        return model, metrics_train, metrics_test

    def train_ada_boost(self):
        model = AdaBoostClassifier(
            n_estimators=200,
            learning_rate=0.5,
            random_state=42
        )
        model.fit(self.X_train, self.Y_train)
        prob_train = model.predict_proba(self.X_train)[:, 1]
        prob_test = model.predict_proba(self.X_test)[:, 1]
        metrics_train = self.compute_metrics(self.Y_train, prob_train)
        metrics_test = self.compute_metrics(self.Y_test, prob_test)
        return model, metrics_train, metrics_test

    def train_naive_bayes(self):
        model = GaussianNB()
        model.fit(self.X_train, self.Y_train)
        prob_train = model.predict_proba(self.X_train)[:, 1]
        prob_test = model.predict_proba(self.X_test)[:, 1]
        metrics_train = self.compute_metrics(self.Y_train, prob_train)
        metrics_test = self.compute_metrics(self.Y_test, prob_test)
        return model, metrics_train, metrics_test

    def train_svm(self):
        model = SVC(
            kernel="rbf",
            C=1,
            probability=True,
            random_state=42
        )
        model.fit(self.X_train, self.Y_train)
        prob_train = model.predict_proba(self.X_train)[:, 1]
        prob_test = model.predict_proba(self.X_test)[:, 1]
        metrics_train = self.compute_metrics(self.Y_train, prob_train)
        metrics_test = self.compute_metrics(self.Y_test, prob_test)
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
            random_state=42,
            verbosity=0
        )
        model.fit(self.X_train, self.Y_train)
        prob_train = model.predict_proba(self.X_train)[:, 1]
        prob_test = model.predict_proba(self.X_test)[:, 1]
        metrics_train = self.compute_metrics(self.Y_train, prob_train)
        metrics_test = self.compute_metrics(self.Y_test, prob_test)
        self.explainability(model)
        return model, metrics_train, metrics_test