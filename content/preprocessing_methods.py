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
        numeric_columns = self.X.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = self.X.select_dtypes(include=['object', 'category']).columns.tolist()
        missing_percentage = self.X_train.isnull().mean()
        threshold_knn = 0.2
        threshold_iterative = 0.5
        threshold_categorical_to_drop = 0.2
        knn_features = [col for col in numeric_columns if missing_percentage[col] <= threshold_knn]
        iterative_features = [col for col in numeric_columns if threshold_knn < missing_percentage[col] <= threshold_iterative]
        drop_features = [col for col in numeric_columns if missing_percentage[col] > threshold_iterative]
        drop_features_categorical = [col for col in categorical_columns if missing_percentage[col] > threshold_categorical_to_drop]
        scaler_knn = StandardScaler()
        X_train_knn_scaled = scaler_knn.fit_transform(self.X_train[knn_features])
        X_test_knn_scaled = scaler_knn.transform(self.X_test[knn_features])
        knn_imputer = KNNImputer(n_neighbors=10)
        self.X_train[knn_features] = scaler_knn.inverse_transform(knn_imputer.fit_transform(X_train_knn_scaled))
        self.X_test[knn_features] = scaler_knn.inverse_transform(knn_imputer.transform(X_test_knn_scaled))
        iterative_imputer = IterativeImputer(max_iter=10, random_state=self.seed)
        self.X_train[iterative_features] = iterative_imputer.fit_transform(self.X_train[iterative_features])
        self.X_test[iterative_features] = iterative_imputer.transform(self.X_test[iterative_features])
        cat_imputer = SimpleImputer(strategy='most_frequent')
        self.X_train[categorical_columns] = cat_imputer.fit_transform(self.X_train[categorical_columns])
        self.X_test[categorical_columns] = cat_imputer.transform(self.X_test[categorical_columns])
        all_drop_features = drop_features + drop_features_categorical
        self.X_train.drop(columns=all_drop_features, inplace=True)
        self.X_test.drop(columns=all_drop_features, inplace=True)
        self.original_X_train = self.X_train.copy()
        self.original_X_test = self.X_test.copy()
        numeric_columns = self.X_train.select_dtypes(include=['number']).columns.tolist()
        self.scaler = StandardScaler()
        self.X_train[numeric_columns] = self.scaler.fit_transform(self.X_train[numeric_columns])
        self.X_test[numeric_columns] = self.scaler.transform(self.X_test[numeric_columns])
        self.transformer = ColumnTransformer(
            transformers=[('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_columns)],
            remainder='passthrough'
        )
        self.X_train = self.transformer.fit_transform(self.X_train)
        self.X_test = self.transformer.transform(self.X_test)
        ohe = self.transformer.named_transformers_['cat']
        ohe_names = ohe.get_feature_names_out(categorical_columns)
        final_numeric = self.original_X_train.select_dtypes(include=['number']).columns.tolist()
        final_numeric = [col for col in final_numeric if col not in all_drop_features]
        self.final_feature_names = list(ohe_names) + final_numeric
        self.X_train = pd.DataFrame(self.X_train, columns=self.final_feature_names)
        self.X_test = pd.DataFrame(self.X_test, columns=self.final_feature_names)
        print("Preprocessing complete.")
        return self.X_train, self.X_test, self.final_feature_names, self.original_X_train, self.original_X_test