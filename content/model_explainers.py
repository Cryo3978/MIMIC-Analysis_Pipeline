# model_explainers.py

import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

class ModelExplainer:
    def __init__(self, model, X_train, X_test, feature_names=None):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names or list(X_train.columns)

    def explain_shap(self, plot_summary=True, plot_force=False):
        print("\n[SHAP Explanation]")
        try:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(self.X_test)

            if plot_summary:
                shap.summary_plot(shap_values, self.X_test, feature_names=self.feature_names)

            if plot_force:
                shap.initjs()
                shap.force_plot(explainer.expected_value, shap_values[0, :], self.X_test.iloc[0, :])

        except Exception as e:
            print("SHAP explanation failed:", str(e))

    def explain_lime(self, instance_idx=0, num_features=10):
        print("\n[LIME Explanation]")
        try:
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=self.X_train.values,
                feature_names=self.feature_names,
                class_names=["class_0", "class_1"],
                mode="classification"
            )

            exp = explainer.explain_instance(
                data_row=self.X_test.iloc[instance_idx].values,
                predict_fn=self.model.predict_proba,
                num_features=num_features
            )

            exp.show_in_notebook(show_table=True)

        except Exception as e:
            print("LIME explanation failed:", str(e))
