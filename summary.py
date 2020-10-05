"""Calculate Rt given a posterior"""
import argparse
import os
import yaml
import h5py
import numpy as np
import pandas as pd
import geopandas as gp

import tensorflow as tf

from covid.cli_arg_parse import cli_args
from covid.model import (
    rayleigh_quotient,
    power_iteration,
)
from covid.impl.util import compute_state
from covid.summary import mean_and_ci

import model_spec

DTYPE = model_spec.DTYPE

GIS_TEMPLATE = "data/UK2019mod_pop.gpkg"

# Reproduction number calculation
def calc_R_it(theta, xi, events, init_state, covar_data):
    """Calculates effective reproduction number for batches of metapopulations
    :param theta: a tensor of batched theta parameters [B] + theta.shape
    :param xi: a tensor of batched xi parameters [B] + xi.shape
    :param events: a [B, M, T, X] batched events tensor
    :param init_state: the initial state of the epidemic at earliest inference date
    :param covar_data: the covariate data
    :return a batched vector of R_it estimates
    """
    print("Theta shape: ", theta.shape)

    def r_fn(args):
        theta_, xi_, events_ = args
        t = events_.shape[-2] - 1
        state = compute_state(init_state, events_, model_spec.STOICHIOMETRY)
        state = tf.gather(state, t - 1, axis=-2)  # State on final inference day

        par = dict(beta1=theta_[0], beta2=theta_[1], gamma=theta_[2], xi=xi_)

        ngm_fn = model_spec.next_generation_matrix_fn(covar_data, par)
        ngm = ngm_fn(t, state)
        return ngm

    return tf.vectorized_map(r_fn, elems=(theta, xi, events))


@tf.function
def predicted_incidence(theta, xi, init_state, init_step, num_steps):
    """Runs the simulation forward in time from `init_state` at time `init_time`
       for `num_steps`.
    :param theta: a tensor of batched theta parameters [B] + theta.shape
    :param xi: a tensor of batched xi parameters [B] + xi.shape
    :param events: a [B, M, S] batched state tensor
    :param init_step: the initial time step
    :param num_steps: the number of steps to simulate
    :returns: a tensor of srt_quhape [B, M, num_steps, X] where X is the number of state 
              transitions
    """

    def sim_fn(args):
        theta_, xi_, init_ = args

        par = dict(beta1=theta_[0], beta2=theta_[1], gamma=theta_[2], xi=xi_)

        model = model_spec.CovidUK(
            covar_data,
            initial_state=init_,
            initial_step=init_step,
            num_steps=num_steps,
        )
        sim = model.sample(**par)
        return sim["seir"]

    events = tf.map_fn(
        sim_fn, elems=(theta, xi, init_state), fn_output_signature=(tf.float64),
    )
    return events


# Today's prevalence
def prevalence(predicted_state, population_size, name=None):
    """Computes prevalence of E and I individuals

    :param state: the state at a particular timepoint [batch, M, S]
    :param population_size: the size of the population
    :returns: a dict of mean and 95% credibility intervals for prevalence
              in units of infections per person
    """
    prev = tf.reduce_sum(predicted_state[:, :, 1:3], axis=-1) / tf.squeeze(
        population_size
    )
    return mean_and_ci(prev, name=name)


def predicted_events(events, name=None):
    num_events = tf.reduce_sum(events, axis=-1)
    return mean_and_ci(num_events, name=name)


if __name__ == "__main__":

    args = cli_args()

    # Get general config
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    inference_period = [
        np.datetime64(x) for x in config["settings"]["inference_period"]
    ]

    # Load covariate data
    covar_data = model_spec.read_covariates(
        config["data"], date_low=inference_period[0], date_high=inference_period[1]
    )

    # Load posterior file
    posterior = h5py.File(
        os.path.expandvars(
            os.path.join(config["output"]["results_dir"], config["output"]["posterior"])
        ),
        "r",
        rdcc_nbytes=1024 ** 3,
        rdcc_nslots=1e6,
    )

    # Pre-determined thinning of posterior (better done in MCMC?)
    idx = range(6000, 10000, 10)
    theta = posterior["samples/theta"][idx]
    xi = posterior["samples/xi"][idx]
    events = posterior["samples/events"][idx]
    init_state = posterior["initial_state"][:]
    state_timeseries = compute_state(init_state, events, model_spec.STOICHIOMETRY)

    # Build model
    model = model_spec.CovidUK(
        covar_data, initial_state=init_state, initial_step=0, num_steps=events.shape[1],
    )

    ngms = calc_R_it(theta, xi, events, init_state, covar_data)
    b, _ = power_iteration(ngms)
    rt = rayleigh_quotient(ngms, b)
    q = np.arange(0.05, 1.0, 0.05)
    rt_quantiles = pd.DataFrame({"Rt": np.quantile(rt, q)}, index=q).T.to_excel(
        os.path.join(config["output"]["results_dir"], config["output"]["national_rt"]),
    )

    # Prediction requires simulation from the last available timepoint for 28 + 4 + 1 days
    # Note a 4 day recording lag in the case timeseries data requires that
    # now = state_timeseries.shape[-2] + 4
    prediction = predicted_incidence(
        theta,
        xi,
        init_state=state_timeseries[..., -1, :],
        init_step=state_timeseries.shape[-2] - 1,
        num_steps=33,
    )
    predicted_state = compute_state(
        state_timeseries[..., -1, :], prediction, model_spec.STOICHIOMETRY
    )

    # Prevalence now
    prev_now = prevalence(predicted_state[..., 4, :], covar_data["N"], name="prev")

    # Incidence of detections now
    cases_now = predicted_events(prediction[..., 4:5, 2], name="cases")

    # Incidence from now to now+7
    cases_7 = predicted_events(prediction[..., 4:11, 2], name="cases7")
    cases_14 = predicted_events(prediction[..., 4:18, 2], name="cases14")
    cases_21 = predicted_events(prediction[..., 4:25, 2], name="cases21")
    cases_28 = predicted_events(prediction[..., 4:32, 2], name="cases28")

    # Prevalence at day 7
    prev_7 = prevalence(predicted_state[..., 11, :], covar_data["N"], name="prev7")
    prev_14 = prevalence(predicted_state[..., 18, :], covar_data["N"], name="prev14")
    prev_21 = prevalence(predicted_state[..., 25, :], covar_data["N"], name="prev21")
    prev_28 = prevalence(predicted_state[..., 28, :], covar_data["N"], name="prev28")

    def geosummary(geodata, summaries):
        for summary in summaries:
            for k, v in summary.items():
                arr = v
                if isinstance(v, tf.Tensor):
                    arr = v.numpy()
                geodata[k] = arr

    ## GIS here
    ltla = gp.read_file(GIS_TEMPLATE, layer="UK2019mod_pop_xgen")
    ltla = ltla[ltla["lad19cd"].str.startswith("E")]  # England only, for now.
    ltla = ltla.sort_values("lad19cd")
    rti = tf.reduce_sum(ngms, axis=-1)

    geosummary(
        ltla,
        (
            mean_and_ci(rti, name="Rt"),
            prev_now,
            cases_now,
            prev_7,
            prev_14,
            prev_21,
            prev_28,
            cases_7,
            cases_14,
            cases_21,
            cases_28,
        ),
    )

    ltla["Rt_exceed"] = np.mean(rti > 1.0, axis=0)
    ltla = ltla.loc[
        :,
        ltla.columns.str.contains(
            "(lad19cd|lad19nm$|prev|cases|Rt|popsize|geometry)", regex=True
        ),
    ]
    ltla.to_file(
        os.path.join(config["output"]["results_dir"], config["output"]["geopackage"]),
        driver="GPKG",
    )
