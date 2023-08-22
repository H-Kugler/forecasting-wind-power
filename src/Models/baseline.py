import pandas as pd
from typing import Literal

from src.Models.basemodel import Basemodel
from src.DataHandling.processing import supervised_transform


class Baseline(Basemodel):

    def __init__(self, data: pd.DataFrame, horizon: Literal["10min", "Hourly", "Daily"]):
        """
        Initializes the model.
        :param data: The data to fit the model to
        :param horizon: The horizon to predict on
        """
        super(Baseline, self).__init__(data, horizon, window_size=1)

    def fit(self, test_start, test_end):
        """
        Fits the model to the given data.
        :param data: The data to fit the model to
        """
        X, y = supervised_transform(
            df=self.data,
            time_steps_ahead=self.time_steps_ahead,
            window_size=self.window_size,
        )
        self.X_test = X.loc[test_start:test_end]
        self.y_test = y.loc[test_start:test_end]

    def predict(self):
        """
        Predicts the next Power output time_steps_ahead into the future.
        :return: A numpy array of predictions
        """
        if self.X_test is None:
            raise ValueError("Model has not been fitted yet.")
        return self.X_test[f"Power (kW) (time -{self.time_steps_ahead})"].values
