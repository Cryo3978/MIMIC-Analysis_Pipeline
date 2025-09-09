import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from content.feature_methods import FeatureMethods
from content.model_training import ModelTrainer
from content.model_explainers import ModelExplainer
from content.preprocessing_methods import PreprocessingMethods
import tensorflow as tf


class MIMICPipeline:
    def __init__(self, path,y_position=6, seed=42, test_size=0.2):
        self.path = path
        self.seed = seed
        self.y_position=y_position
        self.test_size=test_size
        self._set_seed()
        self._load_data()
        self._split_data()
        self._preprocess()
        self._feature_engineering()
        self._model_training()
        self._explainability()
    
    def _set_seed(self):
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        os.environ['TF_NUM_INTEROP_THREADS'] = '1'
        os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
        
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def _load_data(self,y_position=7):
        self.data = pd.read_excel(self.path)
        self.X = self.data.iloc[:, y_position+1:]
        self.Y = self.data.iloc[:, y_position]
        print("Data loaded successfully")

    def _split_data(self):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X, self.Y, test_size=0.2, random_state=self.seed
        )

    def _preprocess(self):
        preprocessor = PreprocessingMethods(self.X_train, self.X_test, self.X.copy(), seed=self.seed)
        self.X_train, self.X_test, self.final_feature_names, self.original_X_train, self.original_X_test = preprocessor.run()

    def _feature_engineering(self):
        fm = FeatureMethods(self.X_train, self.Y_train, self.X_test)

        # VIF
        high_vif_features, self.vif_df = fm.calculate_vif(threshold=10, plot=True)
        self.X_train = self.X_train.drop(columns=high_vif_features)
        self.X_test = self.X_test.drop(columns=high_vif_features)

        # MI
        self.selected_mi_features, self.mi_df = fm.calculate_mutual_info(threshold=0.015, plot=True)

        # RFE
        self.X_train, self.X_test, self.final_selected_features= fm.rfe_selection(num_features=24)


    def _model_training(self):
        trainer = ModelTrainer(self.X_train, self.Y_train, self.X_test, self.Y_test)

        self.model_lr, auc_lr, self.cm_lr = trainer.train_logistic_regression()
        print("LR AUC:", auc_lr)

        self.model_xgb, auc_xgb, self.cm_xgb = trainer.train_xgboost()
        print("XGB AUC:", auc_xgb)

        self.model_nn, auc_nn, self.cm_nn = trainer.train_neural_network()
        print("NN AUC:", auc_nn)
        # And more

    def _explainability(self):
        model = XGBClassifier().fit(self.X_train, self.Y_train)

        explainer = ModelExplainer(
            model=model,
            X_train=self.X_train,
            X_test=self.X_test,
            feature_names=self.final_selected_features
        )

        explainer.explain_shap(plot_summary=True)
        explainer.explain_lime(instance_idx=0)


if __name__ == "__main__":
    pipeline = MIMICPipeline(path="./datasets/COPD.xlsx", y_position=7)
