from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
from typing import Literal, Union, List, Tuple


def train_test_split(
    test_start: str, test_end: str, df: pd.DataFrame, target_var: str
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Splits the given data into train and test set.
    :param test_start: The start of the test set - has to be a string parsable by pd.to_datetime
    :param test_end: The end of the test set - has to be a string parsable by pd.to_datetime
    :param df: The data to split
    :return: The train and test set
    """
    test_start = pd.to_datetime(test_start)
    test_end = pd.to_datetime(test_end)

    # check if test_start and test_end are in data
    assert test_start in df.index, "test_start not in data."
    assert test_end in df.index, "test_end not in data."
    # check if test_start is before test_end
    assert test_start < test_end, "test_start must be before test_end."

    # check if target_var is in data
    assert target_var in df.columns, "target_var not in data."

    X_train = df.loc[:test_start,][:-1]
    y_train = df.loc[:test_start][target_var][:-1]
    X_test = df.loc[test_start:test_end,]
    y_test = df.loc[test_start:test_end][target_var]

    return X_train, y_train, X_test, y_test


class SupervisedTransformer(BaseEstimator, TransformerMixin):
    """
    Encodes time series data (i.e. index is pd.datetime) into a supervised learning problem.
    """

    def __init__(
        self,
        horizon: Literal[1, 6, 144] = 1,
        window_size: int = 1,
        encode_time: List[Literal["hour", "dayofweek", "month", "year"]] = None,
        include_past: bool = True,
    ):
        """
        Initializes the transformer.
        :param horizon: Specifies the time horizon of the prediction: - 10min: 1 time step ahead
                                                                      - Hourly: 6 time steps ahead
                                                                      - Daily: 144 time steps ahead
        :param window_size: The number of time steps into the past to use for prediction
        :param encode_time: Whether to encode the time features
        :param include_past: Whether to include the past observations of the target in the prediction
        """
        self.horizon = horizon
        self.window_size = window_size
        self.encode_time = encode_time
        self.include_past = include_past

    def _reset(self):
        """
        Resets the transformer.
        """
        if hasattr(self, "X_train"):
            del self.X_train
        if hasattr(self, "target_var"):
            del self.target_var
            del self.target_var_name

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Fits the transformer to the given data.
        :param X: The data to fit the transformer to
        :param y: The target variable
        :return: self
        """
        self._reset()
        assert X.isna().sum().sum() == 0, "There are NaNs in the data."
        if y is not None and self.include_past:
            assert y.name in X.columns, "Target variable not in data."
            self.target_var = y
            self.target_var_name = y

        self.X_train = X.copy()  # has to be stored for transforming the test set

        return self

    def transform(self, X: pd.DataFrame):
        """
        Transforms the given data into a supervised learning problem.
        :param X: The data to transform
        :return: The transformed data
        """
        # check if self is fitted:
        assert hasattr(self, "X_train"), "Transformer has to be fitted first."
        # check if there are NaNs in the data
        assert X.isna().sum().sum() == 0, "There are NaNs in the data."

        data = X.copy()

        if self.encode_time is not None:
            for time_feature in self.encode_time:
                data[time_feature] = self._append_time_feature(data.index, time_feature)

        if self.X_train.index.equals(data.index):
            return self._supervised_transform(data)
        else:
            return self._supervised_transform_test(data)

    def _supervised_transform(self, X: pd.DataFrame):
        """
        Transforms the given data into a supervised learning problem.
        :param X: The data to transform
        :param y: The target variable
        :return: The transformed data
        """

        shifts = range(self.horizon, self.window_size + self.horizon)
        predictors = X.columns

        for column in predictors:
            for i in shifts:
                X[f"{column} (time {-i})"] = X[column].shift(i)

        X.dropna(inplace=True)
        # drop columns of current timestep
        X.drop(columns=predictors, inplace=True)

        return X

    def _supervised_transform_test(self, X: pd.DataFrame):
        """
        Transform the given data into a supervised learning problem for the test set.
        :param X: The data to transform (the test set must be directly after the train set)
        :return: The transformed data
        """
        # check if index of X is directly after index of X_train
        #  assert X.index[0] == self.X_train.index[-1] + pd.Timedelta(self.horizon, unit="min"), "Index of test set must be directly after index of train set."

        # append the last window size observations of the train set to the test set
        X = pd.concat([self.X_train.iloc[-(self.window_size + self.horizon - 1) :], X])

        return self._supervised_transform(X)

    def _append_time_feature(
        dates: pd.DatetimeIndex, feature: Literal["dayofweek", "month", "year"]
    ) -> pd.Series:
        pass


class DataCleaner(BaseEstimator, TransformerMixin):
    """
    Cleans the given data.
    Essentially, this transformer performs all steps that are described in the data inspection notebook.
    """

    def __init__(self, verbose: bool = False):
        """
        Initializes the transformer.
        :param verbose: Whether to print information about the NaNs
        """
        self.verbose = verbose

    def _reset(self):
        """
        Resets the transformer.
        """
        if hasattr(self, "features"):
            del self.features
        if hasattr(self, "target_var"):
            del self.target_var

    def fit(self, X: pd.DataFrame, y: pd.Series = None, features: List[str] = None):
        """
        Fits the transformer to the given data.
        X: The data to fit the transformer to
        """
        # check if features are columns in data frame
        self._reset()

        if features is not None:
            assert all(
                feature in X.columns for feature in features
            ), "Features not in data. Please select features from: " + str(X.columns)
            self.features = features
        else:
            self.features = X.columns

        if y is not None:
            self.target_var = y.name

        ### TODO: Perform further sanity checks on the data ? ###

        return self

    def transform(
        self, X: pd.DataFrame, y: pd.Series = None, renamed_features: List[str] = None
    ):
        """
        Cleans the given data.
        :param X: The data to clean
        :return: The cleaned data
        """
        # check if features are columns in data frame
        assert all(
            feature in X.columns for feature in self.features
        ), "Features not in data. Please select valid data or fit the transformer first."

        X = X[self.features]

        if renamed_features is not None:
            assert len(self.features) == len(
                renamed_features
            ), "Number of features and number of renamed features must be equal."
            X.columns = renamed_features
            self.features = renamed_features

        # append

        return self._clean_data(X)

    def _clean_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the given data.
        :param X: The data to clean
        :return: The cleaned data
        """
        # Eliminate columns with to many nans (e.g. 50%)
        if not hasattr(self, "features"):
            X.dropna(axis=1, thresh=0.5 * X.shape[0], inplace=True)
            self.features = X.columns

        pass


class Scaler(StandardScaler):
    """
    Scales the given data.
    """

    def fit(
        self,
        X,
        y = None,
        sample_weight=None,
    ):
        return super().fit(X, y, sample_weight)

    def transform(
        X,
        y = None,
        copy: bool = True,
    ):
        # TODO: Implement this method
        pass
