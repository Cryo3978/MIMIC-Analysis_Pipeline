# model_training.py

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
import lightgbm as lgb
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

class ModelTrainer:
    def __init__(self, X_train, Y_train, X_test, Y_test):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

    def train_logistic_regression(self):
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(self.X_train, self.Y_train)
        pred_test = model.predict(self.X_test)
        auc_test = roc_auc_score(self.Y_test, pred_test)
        return model, auc_test, confusion_matrix(self.Y_test, pred_test)

    def train_xgboost(self):
        model = XGBClassifier(
            objective="binary:logistic",
            n_estimators=1000,
            learning_rate=0.025,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.05,
            reg_lambda=0.08,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42
        )
        model.fit(self.X_train, self.Y_train, eval_set=[(self.X_test, self.Y_test)], verbose=False)
        prob_test = model.predict_proba(self.X_test)[:, 1]
        auc_test = roc_auc_score(self.Y_test, prob_test)
        return model, auc_test, confusion_matrix(self.Y_test, prob_test > 0.5)

    def train_neural_network(self):
        model = Sequential([
            Dense(128, activation="relu", input_dim=self.X_train.shape[1], kernel_regularizer=l2(0.01)),
            Dense(64, activation="relu", kernel_regularizer=l2(0.01)),
            Dense(16, activation="relu"),
            Dense(8, activation="relu"),
            Dense(1, activation="sigmoid")
        ])
        model.compile(optimizer=Adam(learning_rate=0.015), loss="binary_crossentropy", metrics=["AUC"])
        es = EarlyStopping(monitor="val_loss", patience=10, min_delta=1e-4, restore_best_weights=True)
        model.fit(self.X_train, self.Y_train, validation_data=(self.X_test, self.Y_test),
                  epochs=1000, batch_size=256, callbacks=[es], verbose=0)
        prob_test = model.predict(self.X_test)[:, 0]
        auc_test = roc_auc_score(self.Y_test, prob_test)
        return model, auc_test, confusion_matrix(self.Y_test, prob_test > 0.5)

    def train_random_forest(self):
        model = RandomForestRegressor(
            n_estimators=1000, criterion='squared_error',
            max_depth=7, min_samples_split=20, random_state=42
        )
        model.fit(self.X_train, self.Y_train)
        pred_test = model.predict(self.X_test)
        auc_test = roc_auc_score(self.Y_test, pred_test)
        return model, auc_test, confusion_matrix(self.Y_test, pred_test > 0.25)

    def train_ada_boost(self):
        model = AdaBoostRegressor(
            n_estimators=200, learning_rate=0.5,
            loss='linear', random_state=42
        )
        model.fit(self.X_train, self.Y_train)
        pred_test = model.predict(self.X_test)
        auc_test = roc_auc_score(self.Y_test, pred_test)
        return model, auc_test, confusion_matrix(self.Y_test, pred_test > 0.25)

    def train_naive_bayes(self):
        model = GaussianNB()
        model.fit(self.X_train, self.Y_train)
        pred_test = model.predict(self.X_test)
        auc_test = roc_auc_score(self.Y_test, pred_test)
        return model, auc_test, confusion_matrix(self.Y_test, pred_test)

    def train_svm(self):
        model = SVR(kernel='rbf', C=1, epsilon=0.03)
        model.fit(self.X_train, self.Y_train)
        pred_test = model.predict(self.X_test)
        auc_test = roc_auc_score(self.Y_test, pred_test)
        return model, auc_test, confusion_matrix(self.Y_test, pred_test > 0.25)

    def train_lightgbm(self):
        model = lgb.LGBMRegressor(
            objective='regression',
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            colsample_bytree=0.8,
            subsample=0.8,
            reg_alpha=0.8,
            reg_lambda=0.7,
            random_state=42
        )
        model.fit(self.X_train, self.Y_train)
        pred_test = model.predict(self.X_test)
        auc_test = roc_auc_score(self.Y_test, pred_test)
        return model, auc_test, confusion_matrix(self.Y_test, pred_test > 0.25)