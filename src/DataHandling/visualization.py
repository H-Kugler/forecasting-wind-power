import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Union, Tuple


def plot_var_over_time(
    column_name: Union[str, List[str]],
    data: pd.DataFrame,
    start_time: str,
    end_time: str,
    save_path: str = None,
):
    """
    Plot a variables over time.
    :param column_name: Name of the columns to plot
    :param data: Dataframe to plot
    :param start_time: Start time of the plot
    :param end_time: End time of the plot
    :param save_path: Path to save the plot
    """
    if isinstance(column_name, str):
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(data.loc[start_time:end_time, column_name])
        ax.set_title(column_name)
        ax.set_xlabel("Time")
        ax.set_ylabel(column_name)
    elif len(column_name) == 1:
        raise ValueError(
            "column_name must be a string if only one column is to be plotted."
        )
    else:
        fig, axs = plt.subplots(
            len(column_name), 1, figsize=(15, 5 * len(column_name)), sharex=True
        )
        for i, col in enumerate(column_name):
            axs[i].plot(data.loc[start_time:end_time, col])
            axs[i].set_title(col)
            axs[i].set_ylabel(col)
            axs[i].set_xticks([])
        axs[-1].set_xlabel("Time")
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def plot_corr_of_vars(
    columns: Union[Tuple[str, str], List[Tuple[str, str]]],
    data: pd.DataFrame,
    save_path: str = None,
):
    """
    Plots a scatter plot of two variables in the data.
    :param columns: List of tuples or tuple of column names to plot against each other
    :param data: Dataframe to plot
    :param save_path: Path to save the plot
    """
    if isinstance(columns, Tuple):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(data[columns[0]], data[columns[1]])
        ax.set_title(f"{columns[0]} vs {columns[1]}")
        ax.set_xlabel(columns[0])
        ax.set_ylabel(columns[1])

    elif len(columns) == 1:
        raise ValueError(
            "columns must be a tuple if only one correlation scatter plot is to be created."
        )
    else:
        fig, axs = plt.subplots(len(columns), 1, figsize=(5, 5 * len(columns)))
        for i, col in enumerate(columns):
            axs[i].scatter(data[col[0]], data[col[1]])
            axs[i].set_title(f"{col[0]} vs {col[1]}")
            axs[i].set_ylabel(col[1])
            axs[i].set_xlabel(col[0])
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

def barplot_errors(df: pd.DataFrame, models: List[str], col: str, save_path: str = None):
    """
    Plots the error of a model.
    :param df: Dataframe containing the error
    :param models: List of models to plot
    :param save_path: Path to save the plot
    """
    df_plot = pd.melt(
        df.reset_index(),
        id_vars=df.index.names,
        value_vars=models,
        var_name="Model",
        value_name="Error",
    )
    if col == "Horizon":
        df_plot["Horizon"] = pd.Categorical(df_plot["Horizon"], categories=["10min", "Hourly", "Daily"], ordered=True)
    sns.catplot(
        x="Metric",
        y="Error",
        hue="Model",
        col=col,
        data=df_plot,
        kind="bar",
        palette="muted",
    )
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def correlationplot(df: pd.DataFrame, save_path: str = None):
    """
    Plots the correlation of a dataframe.
    :param df: Dataframe to plot
    :param save_path: Path to save the plot
    """
    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        fig, ax = plt.subplots()
        ax = sns.heatmap(
            corr, mask=mask, vmax=0.3, square=True, annot=True, cmap="Blues"
        )
        ax.set_title("Correlation Matrix")
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def plot_corr_matrix(data, features):
    corr = data[features].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, ax=ax)
    plt.show()


def scatter_plot(data, features):
    sns.pairplot(data[features], plot_kws={"s": 3})
    plt.show()
