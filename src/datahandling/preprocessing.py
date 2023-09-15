import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from feature_engine.creation import CyclicalFeatures
from typing import Any, Literal, Union, List, Tuple


def train_test_split(
    df: pd.DataFrame, test_start: str, test_end: str, target_var: str
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Splits the given data into train and test set.
    :param df: The data to split
    :param test_start: The start of the test set - has to be a string parsable by pd.to_datetime
    :param test_end: The end of the test set - has to be a string parsable by pd.to_datetime
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
        horizon: Literal["10min", "hourly", "daily"] = "10min",
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

        if self.X_train.index.equals(data.index):
            data = self._supervised_transform(data)
        else:
            data = self._supervised_transform_test(data)

        if self.encode_time is not None:
            for time_feature in self.encode_time:
                data = pd.concat(
                    [
                        data,
                        self._get_time_feature(data.index, time_feature),
                    ],
                    axis=1,
                )
        data.index.name = "Date"
        return data

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
            df = pd.DataFrame(columns=[f"{column} (time {-i})" for i in shifts])
            for i in shifts:
                df[f"{column} (time {-i})"] = X[column].shift(i)
            X = pd.concat([X, df], axis=1)

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

    @staticmethod
    def _get_time_feature(
        dates: pd.DatetimeIndex, feature: Literal["hour", "dayofweek", "month", "year"]
    ) -> pd.Series:
        """
        Transforms the given feature of the given dates into a cyclical feature.
        :param dates: The dates to transform
        :param feature: The feature to transform
        """
        cf = CyclicalFeatures(drop_original=True)
        dates.name = feature
        if feature == "hour":
            return cf.fit_transform(pd.DataFrame(dates.hour, index=dates))
        elif feature == "dayofweek":
            return cf.fit_transform(pd.DataFrame(dates.dayofweek, index=dates))
        elif feature == "month":
            return cf.fit_transform(pd.DataFrame(dates.month, index=dates))
        elif feature == "year":
            return cf.fit_transform(pd.DataFrame(dates.year, index=dates))
        else:
            raise ValueError(
                "feature must be one of 'hour', 'dayofweek', 'month' or 'year'."
            )

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name == "horizon" and isinstance(__value, str):
            if __value == "10min":
                __value = 1
            elif __value == "hourly":
                __value = 6
            elif __value == "daily":
                __value = 144
            else:
                raise ValueError("horizon must be one of '10min', 'hourly' or 'daily'.")
        super().__setattr__(__name, __value)


class DataCleaner(BaseEstimator, TransformerMixin):
    """
    Cleans the given data by interpolating NaNs and removing the remaining NaNs.
    """

    def __init__(
        self,
        features: Union[Literal["auto"], List[str]] = "auto",
        rename_features: Union[Literal[False], List[str]] = False,
    ):
        """
        Initializes the transformer.
        :param features: The features to clean
        :param rename_features: The new names of the features
        :param remove_nans: Whether to remove the remaining NaNs
        """
        self.features = features
        self.rename_features = rename_features

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        This function is called when the transformer is fitted to the data. In this case, it does nothing
        but it is needed for compatibility with sklearn pipelines.
        :param X: The data to fit the transformer to
        :param y: The target variable
        :return: self
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the given data.
        :param X: The data to clean
        :return: The cleaned data
        """
        X = X.copy()
        X = self._select_features(X)
        X = self._clean_data(X)
        return X

    def _clean_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the given data.
        :param X: The data to clean
        :return: The cleaned data
        """
        X = X.interpolate(method="linear")
        X.dropna(inplace=True)
        if isinstance(self.rename_features, list):
            X = self._rename_columns(X, self.rename_features)
        return X

    def _select_features(self, X: pd.DataFrame, thresh: float = 0.7) -> pd.DataFrame:
        """
        Selects the features to clean.
        :param X: The data to select the features from
        :return: Dataframe with the selected features
        """
        assert self._check_features(
            X
        ), "Features not in data. Please select valid data."
        if self.features == "auto":
            self.features = X.columns[X.isna().sum() < thresh * X.shape[0]]
        return X[self.features]

    def _check_features(self, X: pd.DataFrame) -> bool:
        """
        Checks if the features are in the given data.
        :param X: The data to check
        """
        return all(feature in X.columns for feature in self.features)

    def _rename_columns(
        self, X: pd.DataFrame, renamed_features: List[str]
    ) -> pd.DataFrame:
        """
        Renames the columns of the given data.
        :param X: The data to rename
        :param renamed_features: The new names of the features
        :return: The renamed data
        """
        assert len(self.features) == len(
            renamed_features
        ), "Number of features and number of renamed features must be equal."
        X.columns = renamed_features
        return X


class Normalizer(StandardScaler):
    """
    Overwrites the transform function of the StandardScaler from sklearn to allow for to
    keep the index and column names after scaling. Additionally, it allows for no scaling at all such that
    the effects of the transformation can be studied in a pipeline.
    """

    def __init__(self, scale: bool = True):
        """
        Initializes the transformer.
        :param scale: Whether to scale the data
        """
        self.scale = scale
        super().__init__()

    def transform(self, X: pd.DataFrame):
        """
        Transforms the given data.
        :param X: The data to transform
        :return: The transformed data
        """
        if self.scale:
            X = pd.DataFrame(super().transform(X), index=X.index, columns=X.columns)
            return X
        else:
            return X  # return unchanged data


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features: List[str] = ["power"]) -> None:
        """
        Initializes the transformer.
        :param features: The features to select
        """
        self.features = features

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        This function is called when the transformer is fitted to the data. In this case, it does nothing
        but it is needed for compatibility with sklearn pipelines.
        :param X: The data to fit the transformer to
        :param y: The target variable
        :return: self
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Selects the given features from the data.
        :param X: The data to select the features from
        :return: The data with the selected features
        """
        assert self._check_features(
            X
        ), "Features not in data. Please select valid data."
        return X[self.features]

    def _check_features(self, X: pd.DataFrame) -> bool:
        """
        Checks if the features are in the given data.
        :param X: The data to check
        """
        return all(feature in X.columns for feature in self.features)
