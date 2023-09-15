from xgboost import XGBRegressor
import pandas as pd


class XGB(XGBRegressor):
    """
    Overwrites the fit method of the XGBRegressor class to fit the model to the given data.
    This is necessary as the supervised transformer which is in front of the model in our pipeline
    gives transformations of X and y which are not of the same shape and
    therefore not compatible with the fit method of the Ridge class.
    All other methods are inherited from the Ridge class.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the model to the given data.
        :param X: The training data
        :param y: The target variable
        """
        n_samples = X.shape[0]
        y = y.iloc[-n_samples:]
        # check if shapes match
        assert X.shape[0] == y.shape[0]
        # fit model
        super().fit(X, y)
        return self
