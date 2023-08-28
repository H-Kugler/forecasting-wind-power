from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Literal
from sklearn.metrics import mean_squared_error, median_absolute_error

from src.DataHandling.processing import supervised_transform


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

    def fit(self, test_start, test_end):
        """
        Fits the model to the given data.
        :param test_start: The start of the test set
        :param test_end: The end of the test set
        """
        X, y = supervised_transform(
            df=self.data,
            time_steps_ahead=self.time_steps_ahead,
            window_size=self.window_size,
        )
        self.X_train = X.loc[:test_start]
        self.y_train = y.loc[:test_start]
        self.X_test = X.loc[test_start:test_end]
        self.y_test = y.loc[test_start:test_end]

    @abstractmethod
    def predict(self):
        """
        Predicts the next Power output for the given data time_steps_ahead into the future.
        :return: A list-like object of predictions
        """
        pass

    @classmethod
    def evaluate_model_(
        cls,
        datasets: dict,
        test_dates: dict,
        results: dict
    ):
        """
        Evaluates the model on the given datasets.
        :param datasets: A dictionary of datasets to evaluate on
        :param test_dates: A dictionary of test dates to evaluate on
        :param results: A dictionary of results to store the results in
        """
        for dataset_name, dataset in datasets.items():
            for (metric, horizon, window_size), _ in results[dataset_name].iterrows():
                model = cls(
                    data=dataset,
                    horizon=horizon,
                    window_size=window_size
                )
                model.fit(
                    test_start=test_dates[dataset_name][0],
                    test_end=test_dates[dataset_name][1]
                )
                y_pred = model.predict()
                y_true = model.y_test
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mae = median_absolute_error(y_true, y_pred)
                results[dataset_name].loc[(metric, horizon, window_size), cls.__name__] = rmse if metric == "RMSE" else mae