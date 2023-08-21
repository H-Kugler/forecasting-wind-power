import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.kernel_ridge import KernelRidge

from basemodel import Basemodel
from src.DataHandling.processing import supervised_transform


class Regression(Basemodel):
    def __init__(self, lambda_, time_steps_ahead=1, window_size=1):
        self.name = "Ridge Regression"
        self.model = Ridge(alpha=lambda_)
        self.time_steps_ahead = time_steps_ahead
        self.window_size = window_size

    def fit(self, X, y):
        """
        Fits the model to the given data
        :param X: The data to fit on
        :param y: The target values
        :return: None
        """
        X, y = supervised_transform(X, y, self.time_steps_ahead, self.window_size)
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predicts the next Power output for the given data time_steps_ahead into the future.
        :param X: The data to predict on
        :param time_steps_ahead: The number of time steps into the future to predict
        :return: A list of predictions
        """
        X, _ = supervised_transform(X, pd.Series(np.zeros(len(X))), self.time_steps_ahead, self.window_size)
        return self.model.predict(X)
