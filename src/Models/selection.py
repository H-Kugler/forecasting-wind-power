import pandas as pd
from typing import Union
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

from src.Models.basemodel import Basemodel
from src.utils import sort_dict

class GridSearch:
    def __init__(self, model: Union[Pipeline, Basemodel], param_grid: dict):
        """
        Initializes the GridSearch object.
        :param model: The model to use for the grid search
        :param param_grid: The parameter grid to search over
        :param refit: Whether to refit the best model on the whole data
        """
        self.model = model
        self.param_grid = sort_dict(param_grid)
        self.grid = ParameterGrid(self.param_grid)
        self.best_params = None
        self.best_score = None
        self.best_model = None
        self.results = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        refit: bool = True,
    ):
        """
        Performs a grid search over the given parameter.
        :param X_train: The training data
        :param y_train: The training labels
        :param X_test: The test data
        :param y_test: The test labels
        :param refit: Whether to refit the best model
        """

        # create results dataframe
        index = pd.MultiIndex.from_product(
            iterables=list(self.param_grid.values()),
            names=list(self.param_grid.keys()),
        )
        results = pd.DataFrame(index=index, columns=["RMSE", "MAE"]).sort_index()
        for params in self.grid:
            self.model.set_params(**params)
            self.model.fit(X_train, y_train)
            rmse, mae = self.model.score(X_test, y_test)
            results.loc[tuple(params.values()), "RMSE"] = rmse
            results.loc[tuple(params.values()), "MAE"] = mae

        # cast type of RMSE and MAE to float 
        results["RMSE"] = results["RMSE"].astype(float)
        results["MAE"] = results["MAE"].astype(float)    

        self.results = results
        best_params = results["RMSE"].idxmin()
        self.best_score = results.loc[best_params, "RMSE"]
        self.best_params = dict(zip(self.param_grid.keys(), best_params))
        self.best_model = self.model.set_params(**self.best_params)
        if refit:
            self.best_model.fit(X_train, y_train)
