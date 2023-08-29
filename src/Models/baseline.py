import pandas as pd
import numpy as np
from typing import Literal
from sklearn.metrics import mean_squared_error, median_absolute_error

from src.Models.basemodel import Basemodel



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
        :param test_start: The start of the test set
        :param test_end: The end of the test set
        """
        super(Baseline, self).fit(test_start, test_end)

    def predict(self):
        """
        Predicts the next Power output time_steps_ahead into the future.
        :return: A numpy array of predictions
        """
        if self.X_test is None:
            raise ValueError("Model has not been fitted yet.")
        return self.X_test[f"Power (kW) (time -{self.time_steps_ahead})"].values
    
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
        # check if keys are the same for all dictionaries
        if not all(datasets.keys() == test_dates.keys()) or not all(datasets.keys() == results.keys()):
            raise ValueError("Keys of datasets, test_dates and results must be the same.")
        
        for dataset_name, dataset in datasets.items():
            for (metric, horizon, window_size), _ in results[dataset_name].iterrows():
                model = cls(
                    data=dataset,
                    horizon=horizon
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
