import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Union, Tuple


def plot_predictions(
    predictions: pd.DataFrame, y_true: pd.Series = None, save_path: str = None
):
    """
    Plot predictions for each horizon in predictions.
    :param predictions: DataFrame with predictions.
                        Columns are horizons / different models, index is time.
    :param y_true: Series with true values. Index is time.
    :param save_path: Path to save the plot
    """
    horizons = predictions.columns
    _, n = predictions.shape

    # Check if predictions and y_true have the same index
    if y_true is not None:
        assert predictions.index.equals(y_true.index)

    _, axs = plt.subplots(n, 1, figsize=(5 * n, 10), sharex=True)
    for i, horizon in enumerate(horizons):
        if y_true is not None:
            sns.lineplot(
                x=y_true.index,
                y=y_true,
                ax=axs[i],
                label="True",
            )
        sns.lineplot(
            x=predictions.index,
            y=predictions[horizon],
            ax=axs[i],
            label=f"Predictions ({horizon})",
        )
        axs[i].set_ylabel("Power (kW)")
        axs[i].set_title(f"Predictions for horizon '{horizon}'")
        axs[i].legend()

    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def plot_gridsearch_results(
    df: pd.DataFrame,
    x: str,
    hue: str = None,
    col="param_st__horizon",
    benchmarks: list = None,
    save_path: str = None,
):
    """
    Plot the results of a gridsearch in a barplot.
    :param df: Gridsearch results in a dataframe
    :param x: Column to plot on the x-axis
    :param hue: Column to plot as different colors
    :param col: Column to plot as different subplots
    :param benchmarks: List of benchmarks to plot as horizontal lines
    :param save_path: Path to save the plot
    """
    # extract the columns that store the parameters of the model
    params = df.columns[df.columns.str.contains("param_")]
    # extract the columns that store the results of the model
    cols = df.columns[df.columns.str.contains("split")]
    df = -df.set_index(list(params))[cols]
    df = pd.melt(
        df.reset_index(),
        id_vars=params,
        value_vars=cols,
        var_name="split",
        value_name="RMSE",
    )
    g = sns.catplot(
        data=df,
        x=x,
        y="RMSE",
        hue=hue,
        col=col,
        kind="bar",
        palette=sns.color_palette("Set2"),
    )
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle("RMSE for different hyperparameter settings")

    if benchmarks is not None:
        # add horizontal lines with the benchmarks
        for i, ax in enumerate(g.axes.flat):
            ax.axhline(benchmarks[i], ls="--", color="black")
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
