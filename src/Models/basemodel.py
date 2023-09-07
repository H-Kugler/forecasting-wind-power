from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, median_absolute_error
from sklearn.base import BaseEstimator


class Basemodel(ABC, BaseEstimator):
    @abstractmethod
    def fit(self, X, y):
        """
        Fits the model to the given data.
        :param test_start: The start of the test set
        :param test_end: The end of the test set
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predicts the next Power output for the given data time_steps_ahead into the future.
        :return: A list-like object of predictions
        """
        pass

    def score(self, X, y_true):
        """
        Scores the model on the given data.
        :param X: The data to score the model on
        :param y: The target variable
        :return: The rmse and mae score of the prediction
        """
        predictions = self.predict(X)
        rmse = mean_squared_error(y_true, predictions, squared=False)
        mae = median_absolute_error(y_true, predictions)
        return rmse, mae
