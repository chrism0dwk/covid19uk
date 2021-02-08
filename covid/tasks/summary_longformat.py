"""Produces a long-format summary of fitted model results"""

import pickle as pkl
from datetime import date
import numpy as np
import pandas as pd
import xarray

from gemlib.util import compute_state
from covid.model_spec import STOICHIOMETRY
from covid import model_spec
from covid.formats import make_dstl_template


def xarray2summarydf(arr):
    mean = arr.mean(dim="iteration").to_dataset(name="value")
    q = np.arange(start=0.05, stop=1.0, step=0.05)
    quantiles = arr.quantile(q=q, dim="iteration").to_dataset(dim="quantile")
    ds = mean.merge(quantiles).rename_vars({qi: f"{qi:.2f}" for qi in q})
    return ds.to_dataframe().reset_index()


def prevalence(events, popsize):
    prev = compute_state(events.attrs["initial_state"], events, STOICHIOMETRY)
    prev = xarray.DataArray(
        prev.numpy(),
        coords=[
            np.arange(prev.shape[0]),
            events.coords["location"],
            events.coords["time"],
            np.arange(prev.shape[-1]),
        ],
        dims=["iteration", "location", "time", "state"],
    )
    prev_per_1e5 = (
        prev[..., 1:3].sum(dim="state").reset_coords(drop=True)
        / popsize[np.newaxis, :, np.newaxis]
        * 100000
    )
    return xarray2summarydf(prev_per_1e5)


def summary_longformat(input_files, output_file):
    """Draws together pipeline results into a long format
       csv file.

    :param input_files: a list of filenames [data_pkl,
                                             insample14_pkl,
                                             medium_term_pred_pkl,
                                             ngm_pkl]
    :param output_file: the output CSV with columns `[date,
                        location,value_name,value,q0.025,q0.975]`
    """

    with open(input_files[0], "rb") as f:
        data = pkl.load(f)
    da = data["cases"].rename({"date": "time"})
    df = da.to_dataframe(name="value").reset_index()
    df["value_name"] = "newCasesBySpecimenDate"
    df["0.05"] = np.nan
    df["0.5"] = np.nan
    df["0.95"] = np.nan

    # Insample predictive incidence
    with open(input_files[1], "rb") as f:
        insample = pkl.load(f)
    insample_df = xarray2summarydf(insample[..., 2].reset_coords(drop=True))
    insample_df["value_name"] = "insample14_Cases"
    df = pd.concat([df, insample_df], axis="index")

    # Medium term incidence
    with open(input_files[2], "rb") as f:
        medium_term = pkl.load(f)
    medium_df = xarray2summarydf(medium_term[..., 2].reset_coords(drop=True))
    medium_df["value_name"] = "Cases"
    df = pd.concat([df, medium_df], axis="index")

    # Medium term prevalence
    prev_df = prevalence(medium_term, data["N"])
    prev_df["value_name"] = "prevalence"
    df = pd.concat([df, prev_df], axis="index")

    # Rt
    with open(input_files[3], "rb") as f:
        ngms = pkl.load(f)
    rt = ngms.sum(dim="dest")
    rt = rt.rename({"src": "location"})
    rt_summary = xarray2summarydf(rt)
    rt_summary["value_name"] = "R"
    rt_summary["time"] = data["date_range"][1]
    df = pd.concat([df, rt_summary], axis="index")

    quantiles = df.columns[df.columns.str.startswith("0.")]

    return make_dstl_template(
        group="Lancaster",
        model="SpatialStochasticSEIR",
        scenario="Nowcast",
        creation_date=date.today(),
        version=model_spec.VERSION,
        age_band="All",
        geography=df["location"],
        value_date=df["time"],
        value_type=df["value_name"],
        value=df["value"],
        quantiles={q: df[q] for q in quantiles},
    ).to_excel(output_file, index=False)
