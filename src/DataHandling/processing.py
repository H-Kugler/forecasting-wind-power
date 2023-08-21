import pandas as pd
import numpy as np


def supervised_transform(
    df: pd.DataFrame,
    target_var: str = "Power (kW)",
    time_steps_ahead: int = 1,
    window_size: int = 5,
) -> pd.DataFrame:
    """
    Transforms the data into a supervised learning problem.
    :param data: The data to transform
    :param time_steps_ahead: The number of time steps into the future to predict. Default is 1.
    :param window_size: The number of time steps into the past to use for prediction. Default is 5.
    :return: The transformed data
    """
    # check if there are NaNs in the data
    assert df.isna().sum().sum() == 0, "There are NaNs in the data."
    # check if target variable is in data
    assert target_var in df.columns, f"Target variable {target_var} not in data."

    # create copy
    data = df.copy()

    shifts = range(time_steps_ahead, window_size + time_steps_ahead)
    predictors = data.columns
    y = data[target_var]

    for column in predictors:
        for i in shifts:
            data[f"{column} (time {-i})"] = data[column].shift(i)

    # drop columns of current timestep
    data.drop(columns=predictors, inplace=True)
    data.dropna(inplace=True)

    return data, y

# def encode_time(turbine:pd.DataFrame) -> pd.DataFrame:
#     """Encode time features of a turbine."""
#     turbine['month'] = turbine.index.month
#     turbine['hour'] = turbine.index.hour

#     cyclical = CyclicalFeatures(variables=None, drop_original=True)
#     time_features = cyclical.fit_transform(turbine.loc[:, ['month', 'hour']])
#     turbine_encoded = pd.concat([turbine, time_features], axis=1)
#     turbine_encoded.drop(['month',  'hour'], axis=1, inplace=True)
#     return turbine_encoded

def remove_nans(turbine: pd.DataFrame) -> pd.DataFrame:
    """
    Removes NaNs from the data.
    :param turbine: The turbine data
    :return: The cleaned turbine data
    """
    for column in turbine.columns:
        turbine = remove_nans_bae(turbine, column)
    turbine.interpolate(method='linear', inplace=True)
    turbine.fillna(method='bfill', inplace=True)
    turbine.fillna(method='ffill', inplace=True)

    assert turbine.isna().sum().sum() == 0, "There are still NaNs in the data."

    return turbine


def remove_nans_bae(turbine: pd.DataFrame, variable: str, verbose: bool = False) -> pd.DataFrame:
    """
    Removes rows NaNs in variable at the beginning and end of a time series.
    :param turbine: The turbine data
    :param variable: The variable to remove NaNs from
    :return: The cleaned turbine data
    """
    # eliminate NaNs at the beginning of the dataset
    first_valid_row = 0
    for i in range(0, len(turbine)):
        if np.isnan(turbine[variable][i]) or turbine[variable][i] <= 0:
            continue
        else:
            if verbose:
                print(f'This is the first row with a positive value in the target variable {variable}:', i)
            first_valid_row = i
            break

    # do the same for the end of the dataset
    last_valid_row = 0
    for i in range(len(turbine)-1, 0, -1):
        if np.isnan(turbine[variable][i]) or turbine[variable][i] <= 0:
            continue
        else:
            if verbose:
                print(f'This is the last row with a positive value in the target variable {variable}:', i)
            last_valid_row = i
            break
    
    return turbine.iloc[first_valid_row:last_valid_row+1, :]