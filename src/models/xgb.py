from xgboost import XGBRegressor
import pandas as pd


class XGB(XGBRegressor):
    def fit(self, X: pd.DataFrame, y: pd.Series):
        n_samples = X.shape[0]
        y = y.iloc[-n_samples:]
        # check if shapes match
        assert X.shape[0] == y.shape[0]
        # fit model
        super().fit(X, y)
        return self
