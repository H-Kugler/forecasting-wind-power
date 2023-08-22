from typing import Literal
import numpy as np
import pandas as pd

from src.Models.basemodel import Basemodel


class MovingAverage(Basemodel):
    def __init__(
        self,
        data: pd.DataFrame,
        horizon: Literal["10min", "Hourly", "Daily"],
        window_size: int,
        discount: float = 1.0,
    ):
        super().__init__(data, horizon, window_size)
        self.discount = discount

    def fit(self, test_start, test_end):
        """
        Fits the model to the given data.
        :param test_start: The start of the test set
        :param test_end: The end of the test set
        """
        super().fit(test_start, test_end)
        self.X_test = self.X_test[
            [
                f"Power (kW) (time -{i})"
                for i in range(
                    self.time_steps_ahead, self.time_steps_ahead + self.window_size
                )
            ]
        ]
        

    def predict(self):
        """
        Predicts the next Power output for the given data time_steps_ahead into the future.
        :return: A list-like object of predictions
        """
        if self.X_test is None:
            raise ValueError("Model has not been fitted yet.")
        discounts = np.array([self.discount ** i for i in range(self.window_size)])
        predictions = self.X_test.multiply(discounts).sum(axis=1).div(discounts.sum())
        return predictions.to_numpy()
