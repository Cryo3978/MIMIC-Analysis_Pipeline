import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from content.feature_selection import FeatureMethods
from content.model_training import ModelTrainer
from content.preprocessing_methods import PreprocessingMethods
from content.functions import (
    plot_roc_curves,
    plot_calibration_curves_paper,
    plot_decision_curve_analysis,
    compute_threshold_metrics,
)
import tensorflow as tf
import warnings
from sklearn.ensemble import RandomForestClassifier

# display
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", None)
warnings.filterwarnings("ignore", category=FutureWarning)

OUTPUT_DIR = "outputs"


class MIMICPipeline:
    def __init__(self, path, y_position=6, seed=42, test_size=0.2, num_features=8,
                 models_to_run=("lr", "xgb", "rf", "ada", "nb", "svm", "lgbm"), output_dir=None):
        self.path = path
        self.seed = seed
        self.y_position = y_position
        self.test_size = test_size
        self.num_features = num_features
        self.models_to_run = set(models_to_run)
        self.threshold_tables = {}
        self.final_feats = None  # for plotting only
        self.output_dir = output_dir or OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        self._set_seed()
        self._load_data(y_position=y_position)
        self._split_data()
        self._preprocess()
        self._feature_engineering()
        self._model_training()
        self.X_train_original = None
        self.X_test_original = None

        # print summary
        print("-" * 42)
        print(f'num_features:{num_features}')
        print(f"Selected features: {self.final_selected_features}")
        if hasattr(self, "metrics_train_all") and hasattr(self, "metrics_test_all"):
            for name in self.metrics_train_all:
                print(f"Metrics of {name}:")
                print(f"  Train {self.metrics_train_all[name]}")
                print(f"  Test  {self.metrics_test_all[name]}\n")

    def _set_seed(self):
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        os.environ['TF_NUM_INTEROP_THREADS'] = '1'
        os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def _load_data(self, y_position):
        self.data = pd.read_excel(self.path)
        self.X = self.data.iloc[:, y_position + 1:]
        self.Y = self.data.iloc[:, y_position]
        print("Data loaded successfully")

    def _split_data(self):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X, self.Y, test_size=0.2, random_state=self.seed
        )
        self.final_feats = [
            "sofa", "apsiii", "fluid_in_48h_ml", "uo_mlkghr",
            "chemistry_creatinine", "chemistry_sodium", "chemistry_aniongap",
            "rdw", "pt", "ph", "spo2", "resp_rate"
        ]
        self.X_train_original = self.X_train
        self.X_test_original = self.X_test
        fm = FeatureMethods(
            self.X_train_original,
            self.Y_train,
            self.X_test_original,
            seed=self.seed
        )
        smd_df = fm.build_smd_table(
            X_a=self.X_train_original,
            X_b=self.X_test_original,
            features=self.final_feats,
            group_a_name="Training Set",
            group_b_name="Test Set"
        )
        print(smd_df)
        print('\n')
        y_train_series = pd.Series(self.Y_train, index=self.X_train_original.index)
        X_survivors = self.X_train_original.loc[y_train_series == 0]
        X_nonsurvivors = self.X_train_original.loc[y_train_series == 1]
        smd_surv_df = fm.build_smd_table(
            X_a=X_survivors,
            X_b=X_nonsurvivors,
            features=self.final_feats,
            group_a_name="Survivors",
            group_b_name="Non-survivors"
        )
        print(smd_surv_df)

    def _preprocess(self):
        preprocessor = PreprocessingMethods(self.X_train, self.X_test, self.X.copy(), seed=self.seed)
        self.X_train, self.X_test, self.final_feature_names, self.original_X_train, self.original_X_test = preprocessor.run()

    def _feature_engineering(self):
        fm = FeatureMethods(self.X_train, self.Y_train, self.X_test, seed=self.seed)

        self.selected_mi_features, self.mi_df = fm.calculate_mutual_info(
            threshold=0.01,
            plot=True,
            plot_features=self.final_feats
        )
        if len(self.selected_mi_features) > 0:
            self.X_train = self.X_train[self.selected_mi_features]
            self.X_test = self.X_test[self.selected_mi_features]

        self.X_train, self.X_test, self.final_selected_features = fm.rfe_selection(
            num_features=self.num_features,
            estimator=RandomForestClassifier(
                n_estimators=1000,
                max_depth=10,
                random_state=self.seed,
                n_jobs=-1
            )
        )

    def _model_training(self):
        trainer = ModelTrainer(self.X_train, self.Y_train, self.X_test, self.Y_test)

        self.models = {}
        self.metrics_train_all = {}
        self.metrics_test_all = {}

        def _register(model_name, model, m_train, m_test):
            self.models[model_name] = model
            self.metrics_train_all[model_name] = m_train
            self.metrics_test_all[model_name] = m_test

        # Logistic Regression
        if "lr" in self.models_to_run:
            model, m_train, m_test = trainer.train_logistic_regression()
            _register("LogisticRegression", model, m_train, m_test)
            self.model_lr, self.metrics_lr, self.metrics_lr_test = model, m_train, m_test

        # XGBoost
        if "xgb" in self.models_to_run:
            model, m_train, m_test = trainer.train_xgboost()
            _register("XGBoost", model, m_train, m_test)
            self.model_xgb, self.metrics_xgb, self.metrics_xgb_test = model, m_train, m_test

        # Random Forest
        if "rf" in self.models_to_run:
            model, m_train, m_test = trainer.train_random_forest()
            _register("RandomForest", model, m_train, m_test)

        # AdaBoost
        if "ada" in self.models_to_run:
            model, m_train, m_test = trainer.train_ada_boost()
            _register("AdaBoost", model, m_train, m_test)

        # Naive Bayes
        if "nb" in self.models_to_run:
            model, m_train, m_test = trainer.train_naive_bayes()
            _register("GaussianNB", model, m_train, m_test)

        # SVM
        if "svm" in self.models_to_run:
            model, m_train, m_test = trainer.train_svm()
            _register("SVM-RBF", model, m_train, m_test)

        # LightGBM
        if "lgbm" in self.models_to_run:
            try:
                model, m_train, m_test = trainer.train_lightgbm()
                _register("LightGBM", model, m_train, m_test)
            except NameError as e:
                warnings.warn(f"LightGBM not available: {e}")

        self.prob_test_all = {}
        self.prob_train_all = {}

        for name, model in self.models.items():
            prob_train = model.predict_proba(self.X_train)[:, 1]
            prob_test = model.predict_proba(self.X_test)[:, 1]
            self.prob_train_all[name] = prob_train
            self.prob_test_all[name] = prob_test

        plot_roc_curves(
            y_true=self.Y_test,
            prob_dict=self.prob_test_all,
            metrics_dict=self.metrics_test_all,
            title="ROC Curves on Test Set",
            save_path=os.path.join(self.output_dir, "roc_curves_test.png"),
        )

        plot_calibration_curves_paper(
            y_true=self.Y_test,
            prob_dict=self.prob_test_all,
            models_to_plot=["XGBoost", "LogisticRegression", "LightGBM", "RandomForest", "SVM-RBF", "AdaBoost", "GaussianNB"],
            n_bins=5,
            save_path=os.path.join(self.output_dir, "calibration_paper.png"),
        )

        plot_decision_curve_analysis(
            y_true=self.Y_test,
            prob_dict=self.prob_test_all,
            title="Decision Curve Analysis (Test Set)",
            save_path=os.path.join(self.output_dir, "decision_curve_analysis.png"),
        )

        for name, y_prob in self.prob_test_all.items():
            df = compute_threshold_metrics(y_true=self.Y_test, y_prob=y_prob)
            self.threshold_tables[name] = df
            df.to_csv(os.path.join(self.output_dir, f"threshold_metrics_{name}.csv"), index=False)
            print(f"{name} saved.")

if __name__ == "__main__":
    pipeline = MIMICPipeline(path="./datasets/saaki.xlsx", y_position=5,num_features=12)