import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

class FeatureMethods:
    def __init__(self, X_train, Y_train, X_test):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test

    def calculate_vif(self, threshold=10, plot=False):
        vif_data = pd.DataFrame()
        vif_data["feature"] = self.X_train.columns
        vif_data["VIF"] = [
            variance_inflation_factor(self.X_train.values, i)
            for i in range(self.X_train.shape[1])
        ]
        vif_data = vif_data.sort_values("VIF", ascending=False)
        high_vif_features = vif_data[vif_data["VIF"] > threshold]["feature"].tolist()
        if plot:
            import seaborn as sns
            import matplotlib.pyplot as plt
            plt.figure(figsize=(16, 12))
            sns.barplot(x="VIF", y="feature", data=vif_data)
            plt.title("VIF for Features")
            plt.show()
        return high_vif_features, vif_data

    def calculate_mutual_info(self, threshold=0.015, plot=False):
        mi = mutual_info_classif(self.X_train, self.Y_train, random_state=42)
        mi_df = pd.DataFrame({'feature': self.X_train.columns, 'mutual_info': mi})
        mi_df = mi_df.sort_values(by='mutual_info', ascending=False)
        selected_features = mi_df[mi_df['mutual_info'] > threshold]['feature'].tolist()
        if plot:
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.figure(figsize=(12, 8))
            sns.barplot(x='mutual_info', y='feature', data=mi_df, color='lightblue')
            plt.title('Mutual Information')
            plt.show()
        return selected_features, mi_df

    def rfe_selection(self, num_features=24):
        model = RandomForestClassifier(
            n_estimators=100,
            criterion='gini',
            max_depth=10,
            random_state=42
        )
        rfe = RFE(estimator=model, n_features_to_select=num_features)
        rfe.fit(self.X_train, self.Y_train)
        selected_columns = self.X_train.columns[rfe.support_].tolist()
        self.X_train = self.X_train[selected_columns]
        self.X_test = self.X_test[selected_columns]
        return self.X_train, self.X_test, selected_columns