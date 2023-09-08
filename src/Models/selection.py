import pandas as pd
from typing import Union
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from src.models.basemodel import Basemodel
from src.utils import sort_dict


class GridSearch:
    """
    WARNING: This class is deprecated. Use GridSearchCV from sklearn instead
    """

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
        self.best_models = None
        self.results = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        refit: bool = True,
        verbose: bool = False,
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
        results = pd.DataFrame(
            index=index, columns=["RMSE", "MAE"], dtype=float
        ).sort_index()
        for i, params in enumerate(self.grid):
            self.model.set_params(**params)
            self.model.fit(X_train, y_train)
            rmse, mae = self.model.score(X_test, y_test)
            results.loc[tuple(params.values()), "RMSE"] = rmse
            results.loc[tuple(params.values()), "MAE"] = mae
            if i % 3 == 0 and verbose:
                print(f"Finished {i+1} out of {len(self.grid)}")

        self.results = results
        self.best_params = self.results["RMSE"].groupby(level="st__horizon").idxmin()
        self.best_score = self.results.loc[self.best_params, "RMSE"]
        self.best_models = []
        for params in self.best_params:
            model = clone(
                self.model.set_params(**dict(zip(self.param_grid.keys(), params)))
            )
            if refit:
                model.fit(X_train, y_train)
            self.best_models.append(model)

    def update(self, results: pd.DataFrame):
        """
        Updates the gridsearch object with the values in the given results dataframe.
        :param results: The results to update with
        """
        self.results = results
        self.best_params = self.results["RMSE"].groupby(level="st__horizon").idxmin()
        self.best_score = self.results.loc[self.best_params, "RMSE"]
        self.best_models = []
        for params in self.best_params:
            model = clone(
                self.model.set_params(**dict(zip(self.param_grid.keys(), params)))
            )
            self.best_models.append(model)
