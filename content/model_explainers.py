import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
import shap
import lime
import lime.lime_tabular
from content.model_training import compute_metrics

class ModelExplainer:
    def __init__(self, model, X_train, X_test, Y_test, task, feature_names=None):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.feature_names = feature_names or list(X_train.columns)
        self.task = task

    def explain_shap(self, plot_summary=True, plot_force=False, save_force_html=False):
        print("\n[SHAP Explanation]")
        try:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(self.X_test)
            if plot_summary:
                shap.summary_plot(shap_values, self.X_test, feature_names=self.feature_names)
            if plot_force:
                shap.initjs()
                force_plot = shap.force_plot(
                    explainer.expected_value,
                    shap_values[0, :],
                    self.X_test.iloc[0, :]
                )
                if save_force_html:
                    shap.save_html("shap_force_plot.html", force_plot)
        except Exception as e:
            print("SHAP explanation failed:", str(e))

    def explain_lime(self, instance_idx=0, num_features=10, plot=True, save_fig=False):
        print("\n[LIME Explanation]")
        if self.task == 'Mortality':
            class_names = ["Survivors", "Non-Survivors"]
        elif self.task == 'Readmission':
            class_names = ['Non-Readmissions', 'Readmissions']
        try:
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=self.X_train.values,
                feature_names=self.feature_names,
                class_names=class_names,
                mode="classification"
            )
            exp = explainer.explain_instance(
                data_row=self.X_test.iloc[instance_idx].values,
                predict_fn=self.model.predict_proba,
                num_features=num_features
            )
            print(exp.as_list())
            if plot:
                fig = exp.as_pyplot_figure()
                if save_fig:
                    fig.savefig("lime_explanation.png", dpi=300, bbox_inches="tight")
                else:
                    plt.show()
        except Exception as e:
            print("LIME explanation failed:", str(e))

    def ablation_study(self, model, threshold=0.5, plot=True, top_n=20, n_bootstrap=1000, random_state=42):
        rng = np.random.default_rng(random_state)
        results = []
        prob_baseline = model.predict_proba(self.X_test)[:, 1]
        baseline_metrics = compute_metrics(self.Y_test, prob_baseline, threshold, n_bootstrap=n_bootstrap, random_state=random_state)
        baseline_auc = baseline_metrics["AUC"]
        ci_lower, ci_upper = baseline_metrics["AUC_CI95"]
        for col in self.feature_names:
            X_ablate = self.X_test.copy()
            X_ablate[col] = X_ablate[col].mean()
            prob_ablate = model.predict_proba(X_ablate)[:, 1]
            metrics = compute_metrics(self.Y_test, prob_ablate, threshold, n_bootstrap=n_bootstrap, random_state=random_state)
            results.append({
                "Feature": col,
                "AUC": metrics["AUC"],
                "AUC_CI95_lower": metrics["AUC_CI95"][0],
                "AUC_CI95_upper": metrics["AUC_CI95"][1],
                "AUC_drop": baseline_auc - metrics["AUC"]
            })
        df_results = pd.DataFrame(results).sort_values("AUC_drop", ascending=False)
        if plot:
            top_df = df_results.head(top_n)
            plt.figure(figsize=(12, 6))
            plt.bar(
                top_df["Feature"],
                top_df["AUC"],
                yerr=[top_df["AUC"] - top_df["AUC_CI95_lower"], top_df["AUC_CI95_upper"] - top_df["AUC"]],
                color="salmon",
                capsize=5
            )
            plt.axhline(y=baseline_auc, color='blue', linestyle='--', label=f'Baseline AUROC: {baseline_auc:.3f}')
            plt.fill_between(
                range(top_n),
                ci_lower,
                ci_upper,
                color='blue',
                alpha=0.1,
                label='Baseline 95% CI'
            )
            plt.ylabel("AUROC")
            plt.title(f"Ablation Study: Feature Contribution (Top {top_n})")
            plt.xticks(rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            plt.show()
        return df_results