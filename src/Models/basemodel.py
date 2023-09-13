from sklearn.metrics import mean_squared_error, median_absolute_error
from sklearn.base import BaseEstimator


class Basemodel(BaseEstimator):
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
        return -rmse, -mae
