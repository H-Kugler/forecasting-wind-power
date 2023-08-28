from typing import Literal
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, median_absolute_error

from src.Models.basemodel import Basemodel


class Regression(Basemodel):
    def __init__(
        self,
        data: pd.DataFrame,
        horizon: Literal["10min", "Hourly", "Daily"],
        window_size: int,
        model: Literal["linear", "ridge", "lasso", "kernelridge"],
        alpha: float = 1.0,
        gamma: float = 1.0,
    ):
        """
        Initializes the model.
        :param data: The data to fit the model to
        :param horizon: The horizon to predict on
        :param window_size: The number of time steps into the past to use for prediction
        :param model: model type
        :param alpha: regularization parameter
        :param gamma: kernel coefficient
        """
        super().__init__(data, horizon, window_size)
        if model == "linear":
            self.model = LinearRegression()
        elif model == "ridge":
            self.model = Ridge(alpha=alpha)
        elif model == "lasso":
            self.model = Lasso(alpha=alpha)
        elif model == "kernelridge":
            self.model = KernelRidge(alpha=alpha, gamma=gamma)
        else:
            raise ValueError("Invalid model type: " + model)

    def fit(self, test_start, test_end):
        """
        Fits the model to the given data.
        :param test_start: The start of the test set
        :param test_end: The end of the test set
        """
        super().fit(test_start, test_end)
        self.model.fit(self.X_train, self.y_train)
        self.coef_ = self.model.coef_

    def predict(self):
        """
        Predicts the next Power output for the given data time_steps_ahead into the future.
        :return: A list-like object of predictions
        """
        predictions = self.model.predict(self.X_test)
        return predictions
    
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
        for model_name in ["linear", "ridge", "lasso"]:
            for dataset_name, dataset in datasets.items():
                for (metric, horizon, window_size), _ in results[dataset_name].iterrows():
                    model = cls(
                        data=dataset,
                        horizon=horizon,
                        window_size=window_size,
                        model=model_name
                    )
                    model.fit(
                        test_start=test_dates[dataset_name][0],
                        test_end=test_dates[dataset_name][1]
                    )
                    y_pred = model.predict()
                    y_true = model.y_test
                    rmse = mean_squared_error(y_true, y_pred, squared=False)
                    mae = median_absolute_error(y_true, y_pred)
                    results[dataset_name].loc[(metric, horizon, window_size), model_name] = rmse if metric == "RMSE" else mae
