import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer, SimpleImputer
from sklearn.compose import ColumnTransformer

class PreprocessingMethods:
    def __init__(self, X_train, X_test, X, seed=42):
        self.X_train = X_train.copy()
        self.X_test = X_test.copy()
        self.X = X.copy()
        self.seed = seed
        self.final_feature_names = None
        self.scaler = None
        self.transformer = None
        self.original_X_train = None
        self.original_X_test = None

    def run(self):
        print("Starting preprocessing...")
        missing_pct = self.X_train.isnull().mean()
        num_cols = self.X_train.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = self.X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        threshold = 0.2

        drop_cols = [c for c in num_cols + cat_cols if missing_pct[c] > threshold]
        if drop_cols:
            self.X_train = self.X_train.drop(columns=drop_cols)
            self.X_test = self.X_test.drop(columns=drop_cols)

        num_cols = self.X_train.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = self.X_train.select_dtypes(include=["object", "category"]).columns.tolist()

        if num_cols:
            num_imputer = SimpleImputer(strategy="median")
            self.X_train[num_cols] = num_imputer.fit_transform(self.X_train[num_cols])
            self.X_test[num_cols] = num_imputer.transform(self.X_test[num_cols])
        if cat_cols:
            cat_imputer = SimpleImputer(strategy="most_frequent")
            self.X_train[cat_cols] = cat_imputer.fit_transform(self.X_train[cat_cols])
            self.X_test[cat_cols] = cat_imputer.transform(self.X_test[cat_cols])

        self.original_X_train = self.X_train.copy()
        self.original_X_test = self.X_test.copy()

        num_cols = self.X_train.select_dtypes(include=["number"]).columns.tolist()
        if num_cols:
            self.scaler = StandardScaler()
            self.X_train[num_cols] = self.scaler.fit_transform(self.X_train[num_cols])
            self.X_test[num_cols] = self.scaler.transform(self.X_test[num_cols])

        cat_cols = self.X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        transformers = []
        if cat_cols:
            transformers.append(
                ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
            )
        self.transformer = ColumnTransformer(transformers=transformers, remainder="passthrough")

        X_train_arr = self.transformer.fit_transform(self.X_train)
        X_test_arr = self.transformer.transform(self.X_test)

        feature_names = []
        if cat_cols:
            ohe = self.transformer.named_transformers_["cat"]
            feature_names.extend(ohe.get_feature_names_out(cat_cols).tolist())
        feature_names.extend(c for c in self.X_train.columns if c not in cat_cols)
        self.final_feature_names = feature_names

        self.X_train = pd.DataFrame(
            X_train_arr,
            columns=self.final_feature_names,
            index=self.X_train.index
        )

        self.X_test = pd.DataFrame(
            X_test_arr,
            columns=self.final_feature_names,
            index=self.X_test.index
        )

        print("Preprocessing complete.")

        return (
            self.X_train,
            self.X_test,
            self.final_feature_names,
            self.original_X_train,
            self.original_X_test
        )