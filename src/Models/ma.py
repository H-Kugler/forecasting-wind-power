from typing import Union
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator


class MovingAverage(BaseEstimator):
    def __init__(
        self,
        discount: float = 1.0,
    ):
        self.discount = discount

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, str]):
        """
        Fits the model to the given data.
        :param test_start: The start of the test set
        :param test_end: The end of the test set
        """
        if isinstance(y, str):
            self.target_var = y
        else:
            self.target_var = y.name

        # Check if target variable is substring of at least one column in X
        # raise error
        if not X.columns.str.contains(self.target_var, regex=False).any():
            raise ValueError("Target variable not found in X")

    def predict(self, X: pd.DataFrame):
        """
        Predicts the next Power output for the given data time_steps_ahead into the future.
        :return: A list-like object of predictions
        """
        # select columns where target variable is substring of column name
        cols = X.columns[X.columns.str.contains(self.target_var, regex=False)]
        window_size = len(cols)

        if window_size == 0:
            raise ValueError("No columns match target variable")

        # compute predictions
        X = X[cols]
        discounts = np.array([self.discount**i for i in range(window_size)])
        predictions = X.multiply(discounts).sum(axis=1).div(discounts.sum())
        return predictions.to_numpy()
