"""Create insample plots for a given lag"""

import pickle as pkl
import xarray
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")


def plot_timeseries(mean, quantiles, data, dates, title):
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
        dates, y1=quantiles[0], y2=quantiles[-1], color="lightblue", alpha=0.5
    )
    plt.fill_between(
        dates, y1=quantiles[1], y2=quantiles[-2], color="lightblue", alpha=1.0
    )
    plt.plot(dates, mean, color="blue")
    plt.plot(dates, data, "+", color="red")

    plt.title(title)
    fig.autofmt_xdate()

    return fig


def insample_predictive_timeseries(input_files, output_dir, lag):
    """Creates insample plots

    :param input_files: a list of [prediction_file, data_file] (see Details)
    :param output_dir: the output dir to write files to
    :param lag: the number of days at the end of the case timeseries for which to
                plot the in-sample prediction.
    :returns: `None` as output written to disc.

    Details
    -------
    `data_file` is a pickled Python `dict` of data.  It should have a member `cases`
    which is a `xarray` with dimensions [`location`, `date`] giving the number of
    detected cases in each `location` on each `date`.
    `prediction_file` is assumed to be a pickled `xarray` of shape
    `[K,M,T,R]` where `K` is the number of posterior samples, `M` is the number
    of locations, `T` is the number of timepoints, `R` is the number of transitions
    in the model.  The prediction is assumed to start at `cases.coords['date'][-1] - lag`.
    It is assumed that `T >= lag`.

    A timeseries graph (png) summarizing for each `location` the prediction against the
    observations is written to `output_dir`
    """

    prediction_file, data_file = input_files
    lag = int(lag)

    prediction = xarray.open_dataset(prediction_file)["events"]
    prediction = prediction[..., :lag, -1]  # Just removals

    with open(data_file, "rb") as f:
        data = pkl.load(f)

    cases = data["cases"]
    lads = data["locations"]

    # TODO remove legacy code!
    if "lad19cd" in cases.dims:
        cases = cases.rename({"lad19cd": "location"})

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    pred_mean = prediction.mean(dim="iteration")
    pred_quants = prediction.quantile(
        q=[0.025, 0.25, 0.5, 0.75, 0.975],
        dim="iteration",
    )

    for location in cases.coords["location"]:
        print("Location:", location.data)
        fig = plot_timeseries(
            pred_mean.loc[location, :],
            pred_quants.loc[:, location, :],
            cases.loc[location][-lag:],
            cases.coords["date"][-lag:],
            lads.loc[lads["lad19cd"] == location, "name"].iloc[0],
        )
        plt.savefig(output_dir.joinpath(f"{location.data}.png"))
        plt.close()
