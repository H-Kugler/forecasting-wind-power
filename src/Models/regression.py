from typing import Any, Literal

from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.kernel_approximation import Nystroem


class RidgeRegression(Ridge):
    """
    Overwrites the fit method of the Ridge class to fit the model to the given data.
    This is necessary as the supervised transformer which is in front of the model in our pipeline
    gives transformations of X and y which are not of the same shape and
    therefore not compatible with the fit method of the Ridge class.
    All other methods are inherited from the Ridge class.
    """

    def fit(self, X, y):
        """
        Fits the model to the given data.
        :param X: The input data
        :param y: The target data
        """
        n_samples = X.shape[0]
        y = y.iloc[-n_samples:]  # we have to do this because of the rolling window
        # check if shapes match
        assert X.shape[0] == y.shape[0]

        super().fit(X, y)
        return self


class LassoRegression(Lasso):
    """
    Overwrites the fit method of the Lasso class to fit the model to the given data.
    This is necessary as the supervised transformer which is in front of the model in our pipeline
    gives transformations of X and y which are not of the same shape and
    therefore not compatible with the fit method of the Lasso class.
    All other methods are inherited from the Lasso class.
    """

    def fit(self, X, y):
        """
        Fits the model to the given data.
        :param X: The input data
        :param y: The target data
        """
        n_samples = X.shape[0]
        y = y.iloc[-n_samples:]  # we have to do this because of the rolling window
        # check if shapes match
        assert X.shape[0] == y.shape[0]

        super().fit(X, y)
        return self


class Regression:
    """
    WARNING: Deprecated class.
    """

    def __init__(
        self,
        model: Literal["linear", "ridge", "lasso"] = "linear",
        alpha: float = 1.0,
    ):
        """
        Initializes the model.
        :param model: model type
        :param alpha: regularization parameter - useless for model = "linear"
        """
        self.model = model
        self.name = model
        self.alpha = alpha

    def fit(self, X, y):
        """
        Fits the model to the given data.
        :param test_start: The start of the test set
        :param test_end: The end of the test set
        """
        n_samples = X.shape[0]
        y = y.iloc[-n_samples:]
        # check if shapes match
        assert X.shape[0] == y.shape[0]

        self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        Predicts the next Power output for the given data time_steps_ahead into the future.
        :return: A list-like object of predictions
        """
        predictions = self.model.predict(X)
        return predictions

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name == "model" and isinstance(__value, str):
            if __value == "linear":
                super().__setattr__(__name, LinearRegression())
            elif __value == "ridge":
                super().__setattr__(__name, Ridge())
            elif __value == "lasso":
                super().__setattr__(__name, Lasso())
            else:
                raise ValueError("Invalid model type: " + __value)
        elif __name == "alpha" and self.name != "linear" and isinstance(self.name, str):
            self.model.set_params(alpha=__value)
            super().__setattr__(__name, __value)
        else:
            super().__setattr__(__name, __value)
