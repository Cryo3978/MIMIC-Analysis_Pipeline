import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

class FeatureMethods:
    def __init__(self, X_train, Y_train, X_test, seed=42):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.seed = seed

    def calculate_mutual_info(self, threshold=0.015, plot=False, plot_features=None):
        mi = mutual_info_classif(self.X_train, self.Y_train, random_state=self.seed)

        mi_df = pd.DataFrame({
            "feature": self.X_train.columns,
            "mutual_info": mi
        }).sort_values(by="mutual_info", ascending=False)

        selected_features = mi_df[mi_df["mutual_info"] > threshold]["feature"].tolist()

        if plot:
            import matplotlib.pyplot as plt
            import seaborn as sns

            if plot_features is not None:
                plot_feats = [f for f in plot_features if f in mi_df["feature"].values]
                plot_df = mi_df[mi_df["feature"].isin(plot_feats)]
            else:
                plot_df = mi_df

            plt.figure(figsize=(8, max(4, len(plot_df) * 0.4)))
            sns.barplot(
                x="mutual_info",
                y="feature",
                data=plot_df,
                color="lightblue"
            )

            plt.title("Mutual Information (Selected Clinical Features)")
            plt.tight_layout()
            plt.show()

        return selected_features, mi_df

    def rfe_selection(self, num_features=24, estimator=None):
        if estimator is None:
            estimator = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=self.seed,
                n_jobs=-1
            )

        rfe = RFE(estimator=estimator, n_features_to_select=num_features)
        rfe.fit(self.X_train, self.Y_train)

        selected_columns = self.X_train.columns[rfe.support_].tolist()
        self.X_train = self.X_train[selected_columns]
        self.X_test = self.X_test[selected_columns]
        return self.X_train, self.X_test, selected_columns

    @staticmethod
    def _median_iqr(x: pd.Series) -> str:
        """Median [Q1--Q3]."""
        x = pd.to_numeric(x, errors="coerce").dropna()
        if x.empty:
            return "NA"
        q1 = np.percentile(x, 25)
        med = np.percentile(x, 50)
        q3 = np.percentile(x, 75)
        return f"{med:.3f} [{q1:.3f}--{q3:.3f}]"

    @staticmethod
    def _is_binary_series(s: pd.Series) -> bool:
        """True if values in {0, 1}."""
        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.empty:
            return False
        vals = set(pd.unique(s))
        return vals.issubset({0, 1})

    @staticmethod
    def _smd_continuous(x1: pd.Series, x2: pd.Series) -> float:
        """SMD, continuous."""
        x1 = pd.to_numeric(x1, errors="coerce").dropna()
        x2 = pd.to_numeric(x2, errors="coerce").dropna()
        if x1.empty or x2.empty:
            return np.nan

        m1, m2 = x1.mean(), x2.mean()
        s1, s2 = x1.std(ddof=1), x2.std(ddof=1)
        pooled = np.sqrt((s1 ** 2 + s2 ** 2) / 2.0)
        if pooled == 0 or np.isnan(pooled):
            return 0.0
        return (m1 - m2) / pooled

    @staticmethod
    def _smd_binary(x1: pd.Series, x2: pd.Series) -> float:
        """SMD, binary."""
        x1 = pd.to_numeric(x1, errors="coerce").dropna()
        x2 = pd.to_numeric(x2, errors="coerce").dropna()
        if x1.empty or x2.empty:
            return np.nan

        p1 = x1.mean()
        p2 = x2.mean()
        p = (p1 + p2) / 2.0
        denom = np.sqrt(p * (1 - p))
        if denom == 0 or np.isnan(denom):
            return 0.0
        return (p1 - p2) / denom

    def build_smd_table(
        self,
        X_a: pd.DataFrame,
        X_b: pd.DataFrame,
        features: list,
        group_a_name = 'group 1',
        group_b_name = 'group 2'
    ) -> pd.DataFrame:
        """SMD table: median [IQR] per group and |SMD|."""
        rows = []
        for feat in features:
            if feat not in X_a.columns or feat not in X_b.columns:
                continue

            a = X_a[feat]
            b = X_b[feat]

            if self._is_binary_series(pd.concat([a, b], axis=0)):
                smd = self._smd_binary(a, b)
            else:
                smd = self._smd_continuous(a, b)

            rows.append({
                "Feature": feat,
                group_a_name: self._median_iqr(a),
                group_b_name: self._median_iqr(b),
                "SMD": abs(float(smd)) if smd is not None and not np.isnan(smd) else np.nan
            })

        df = pd.DataFrame(rows)
        df = df.sort_values("SMD", ascending=False, na_position="last").reset_index(drop=True)
        return df
