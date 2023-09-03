import pandas as pd
import xarray as xr
from typing import Literal


"""
File to load the data from the two datasets.
Links: - British: https://zenodo.org/record/5841834#.ZEajKXbP2BQ
       - Brazilian: https://zenodo.org/record/1475197#.ZD6iMxXP2WC
"""


def load_data(
    turbine_id: int, which_data: Literal["British", "Brazilian"]
) -> pd.DataFrame:
    """
    Load data from a single turbine.
    :param turbine_id: ID of the turbine to load
    :return: pd.DataFrame with the data of the turbine
    """
    if which_data == "British":
        return _load_british_data(turbine_id)
    elif which_data == "Brazilian":
        return _load_brazilian_data(turbine_id)
    else:
        raise ValueError("which_data must be 'British' or 'Brazilian'")


def _load_british_data(turbine_id: int = 2) -> pd.DataFrame:
    """
    Helper function of load_data to get the data for a single british turbine.
    :param turbine_id: ID of the turbine to load
    :return: pd.DataFrame with the data of the turbine
    """
    _path = "../data/British/"
    path = (
        _path
        + f"Kelmarsh_SCADA_2016_3082/Turbine_Data_Kelmarsh_{turbine_id}_2016-01-03_-_2017-01-01_2{27+turbine_id}.csv"
    )
    turbine_2016 = pd.read_csv(path, header=9)
    path = (
        _path
        + f"Kelmarsh_SCADA_2017_3083/Turbine_Data_Kelmarsh_{turbine_id}_2017-01-01_-_2018-01-01_2{27+turbine_id}.csv"
    )
    turbine_2017 = pd.read_csv(path, header=9)
    path = (
        _path
        + f"Kelmarsh_SCADA_2018_3084/Turbine_Data_Kelmarsh_{turbine_id}_2018-01-01_-_2019-01-01_2{27+turbine_id}.csv"
    )
    turbine_2018 = pd.read_csv(path, header=9)
    path = (
        _path
        + f"Kelmarsh_SCADA_2019_3085/Turbine_Data_Kelmarsh_{turbine_id}_2019-01-01_-_2020-01-01_2{27+turbine_id}.csv"
    )
    turbine_2019 = pd.read_csv(path, header=9)
    path = (
        _path
        + f"Kelmarsh_SCADA_2020_3086/Turbine_Data_Kelmarsh_{turbine_id}_2020-01-01_-_2021-01-01_2{27+turbine_id}.csv"
    )
    turbine_2020 = pd.read_csv(path, header=9)
    path = (
        _path
        + f"Kelmarsh_SCADA_2021_3087/Turbine_Data_Kelmarsh_{turbine_id}_2021-01-01_-_2021-07-01_2{27+turbine_id}.csv"
    )
    turbine_2021 = pd.read_csv(path, header=9)

    # Sanity check (all years have the same columns)
    assert all(turbine_2016.columns == turbine_2017.columns)
    assert all(turbine_2017.columns == turbine_2018.columns)
    assert all(turbine_2018.columns == turbine_2019.columns)
    assert all(turbine_2019.columns == turbine_2020.columns)
    assert all(turbine_2020.columns == turbine_2021.columns)

    # Concatenate all years
    turbine = pd.concat(
        [
            turbine_2016,
            turbine_2017,
            turbine_2018,
            turbine_2019,
            turbine_2020,
            turbine_2021,
        ],
        axis=0,
    )

    # Convert to datetime and set time as index
    # rename the '# Date and time' column to 'Date' for easier access,
    # convert it to pd.datetime and set it as the index
    turbine.rename(columns={"# Date and time": "Date"}, inplace=True)
    turbine["Date"] = pd.to_datetime(turbine["Date"])
    turbine.set_index("Date", inplace=True)

    return turbine


def _load_brazilian_data(turbine_id: int = 2) -> pd.DataFrame:
    """
    Helper function of load_data to get the data for a single brazilian turbine.
    :param turbine_id: ID of the turbine to load
    :return: pd.DataFrame with the data of the turbine
    """
    data = (
        xr.load_dataset("../data/Brazilian/UEBB_v1.nc")
        .sel(Turbine=turbine_id)
        .to_dataframe()
    )

    data = data.loc[data.index.get_level_values("Height") == 100]
    data = data.reset_index(level="Height", drop=True)
    data = data.drop(columns=["Turbine"])

    # Convert to datetime and set time as index
    data.index = pd.to_datetime(data.index)
    data.index.name = "Date"

    return data
