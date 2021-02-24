"""Summary functions"""

import numpy as np
import pickle as pkl
import xarray
import pandas as pd

from gemlib.util import compute_state
from covid.summary import mean_and_ci
from covid.model_spec import STOICHIOMETRY

SUMMARY_DAYS = np.array([1, 7, 14, 21, 28, 35, 42, 49, 56], np.int32)


def rt(input_file, output_file):
    """Reads an array of next generation matrices and
       outputs mean (ci) local Rt values.

    :param input_file: a pickled xarray of NGMs
    :param output_file: a .csv of mean (ci) values
    """

    ngm = xarray.open_dataset(input_file)["ngm"]

    rt = np.sum(ngm, axis=-2)
    rt_summary = mean_and_ci(rt, name="Rt")
    exceed = np.mean(rt > 1.0, axis=0)

    rt_summary = pd.DataFrame(
        rt_summary, index=pd.Index(ngm.coords["dest"], name="location")
    )
    rt_summary["Rt_exceed"] = exceed
    rt_summary.to_csv(output_file)


def infec_incidence(input_file, output_file):
    """Summarises cumulative infection incidence
      as a nowcast, 7, 14, 28, and 56 days.

    :param input_file: a pkl of the medium term prediction
    :param output_file: csv with prediction summaries
    """

    prediction = xarray.open_dataset(input_file)["events"]

    offset = 4
    timepoints = SUMMARY_DAYS + offset

    # Absolute incidence
    def pred_events(events, name=None):
        num_events = np.sum(events, axis=-1)
        return mean_and_ci(num_events, name=name)

    idx = prediction.coords["location"]

    abs_incidence = pd.DataFrame(
        pred_events(prediction[..., offset : (offset + 1), 2], name="cases"),
        index=idx,
    )
    for t in timepoints[1:]:
        tmp = pd.DataFrame(
            pred_events(prediction[..., offset:t, 2], name=f"cases{t-offset}"),
            index=idx,
        )
        abs_incidence = pd.concat([abs_incidence, tmp], axis="columns")

    abs_incidence.to_csv(output_file)


def prevalence(input_files, output_file):
    """Reconstruct predicted prevalence from
       original data and projection.

    :param input_files: a list of [data pickle, prediction netCDF]
    :param output_file: a csv containing prevalence summary
    """
    offset = 4  # Account for recording lag
    timepoints = SUMMARY_DAYS + offset

    with open(input_files[0], "rb") as f:
        data = pkl.load(f)

    prediction = xarray.open_dataset(input_files[1])

    predicted_state = compute_state(
        prediction["initial_state"], prediction["events"], STOICHIOMETRY
    )

    def calc_prev(state, name=None):
        prev = np.sum(state[..., 1:3], axis=-1) / np.squeeze(data["N"])
        return mean_and_ci(prev, name=name)

    idx = prediction.coords["location"]
    prev = pd.DataFrame(
        calc_prev(predicted_state[..., timepoints[0], :], name="prev"),
        index=idx,
    )
    for t in timepoints[1:]:
        tmp = pd.DataFrame(
            calc_prev(predicted_state[..., t, :], name=f"prev{t-offset}"),
            index=idx,
        )
        prev = pd.concat([prev, tmp], axis="columns")

    prev.to_csv(output_file)
