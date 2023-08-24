from xgboost import XGBRegressor
import pandas as pd
from typing import Literal
from src.Models.basemodel import Basemodel


class XGB(Basemodel):
    def __init__(self, data: pd.DataFrame, horizon: Literal["10min", "Hourly", "Daily"], window_size: int):
        """
        Initializes the model.
        :param data: The data to fit the model to
        :param horizon: The horizon to predict on
        """
        super(XGB, self).__init__(data, horizon, window_size)
        self.model = XGBRegressor()

    def fit(self, test_start, test_end):
        """
        Fits the model to the given data.
        :param test_start: The start of the test set
        :param test_end: The end of the test set
        """
        super(XGB, self).fit(test_start, test_end)
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        """
        Predicts the next Power output for the given data time_steps_ahead into the future.
        :return: A list-like object of predictions
        """
        return self.model.predict(self.X_test)
        