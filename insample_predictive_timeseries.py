"""Outputs a predictive timeseries for each LAD"""

import os
from pathlib import Path
import numpy as np
import yaml
import pickle as pkl
import matplotlib.pyplot as plt

from covid.cli_arg_parse import cli_args
from model_spec import read_covariates
from covid.data import read_phe_cases
from covid.data import AreaCodeData


def get_dates(config):
    return [np.datetime64(x) for x in config["Global"]["inference_period"]]


def load_cases(config):
    return read_phe_cases(
        config["data"]["reported_cases"],
        *get_dates(config),
        pillar=config["data"]["pillar"],
        date_type=config["data"]["case_date_type"],
    )


def load_prediction(config):
    prediction_file = os.path.expandvars(
        os.path.join(
            config["output"]["results_dir"],
            config["output"]["insample_prediction"],
        )
    )
    with open(prediction_file, "rb") as f:
        prediction = pkl.load(f)

    return prediction


def plot_timeseries(prediction, data, dates, title):
    """Plots a predictive timeseries with data

    :param prediction: a [5, T]-shaped array with first dimension
                       representing quantiles, and T the number of
                       time points.
    :param data: an array of shape [T] providing the data
    :param dates: an array of shape [T] of type np.datetime64
    :param title: the plot title
    :returns: a matplotlib axis
    """
    fig = plt.figure()

    # In-sample prediction
    plt.fill_between(
        dates, y1=prediction[0], y2=prediction[-1], color="lightblue", alpha=0.5
    )
    plt.fill_between(
        dates, y1=prediction[1], y2=prediction[-2], color="lightblue", alpha=1.0
    )
    plt.plot(dates, prediction[2], color="blue")
    plt.plot(dates, data, "+", color="red")

    plt.title(title)
    fig.autofmt_xdate()
    return fig


def main(config):

    date_low, date_high = get_dates(config)
    cases = load_cases(config)
    prediction = load_prediction(config)[..., -1]  # KxMxTxR
    lads = AreaCodeData.process(config)

    pred_mean = np.mean(prediction, axis=0)
    pred_quants = np.quantile(
        prediction, q=[0.025, 0.25, 0.5, 0.75, 0.975], axis=0
    )
    pred_quants[2] = pred_mean
    dates = np.arange(date_high - 14, date_high)

    results_dir = Path(
        os.path.join(config["output"]["results_dir"], "pred_ts_14day")
    )
    results_dir.mkdir(parents=False, exist_ok=True)
    for i in range(cases.shape[0]):
        title = lads["name"].iloc[i]
        plot_timeseries(
            pred_quants[:, i, :14],
            cases.iloc[i, -14:],
            dates,
            title,
        )
        plt.savefig(results_dir.joinpath(f"{lads['lad19cd'].iloc[i]}.png"))


if __name__ == "__main__":

    args = cli_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Override config file results dir if necessary
    if args.results is not None:
        config["output"]["results_dir"] = args.results

    main(config)
