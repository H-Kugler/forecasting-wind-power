from abc import ABC, abstractmethod
import pandas as pd
from typing import Literal


class Basemodel(ABC):
    TS_10MIN = 1
    TS_1HOUR = 6
    TS_24HOUR = 144

    def __init__(
        self,
        data: pd.DataFrame,
        horizon: Literal["10min", "Hourly", "Daily"],
        window_size: int,
    ):
        """
        Initializes the model.
        :param data: The data to fit the model to
        :param horizon: The horizon to predict on
        :param window_size: The number of time steps into the past to use for prediction
        """
        self.data = data
        self.X_test = None
        self.y_test = None
        if horizon == "10min":
            self.time_steps_ahead = self.TS_10MIN
        elif horizon == "Hourly":
            self.time_steps_ahead = self.TS_1HOUR
        elif horizon == "Daily":
            self.time_steps_ahead = self.TS_24HOUR
        else:
            raise ValueError("Invalid horizon: " + horizon)
        self.window_size = window_size

    @abstractmethod
    def fit(self, test_start, test_end):
        """
        Fits the model to the given data.
        :param data: The data to fit the model to
        """
        pass

    @abstractmethod
    def predict(self):
        """
        Predicts the next Power output for the given data time_steps_ahead into the future.
        :return: A list-like object of predictions
        """
        pass
