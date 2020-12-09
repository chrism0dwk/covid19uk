"""Calculate Rt given a posterior"""
import argparse
import os
import yaml
import h5py
import numpy as np
import pandas as pd
import geopandas as gp

import tensorflow as tf
from gemlib.util import compute_state

from covid.cli_arg_parse import cli_args
from covid.summary import (
    rayleigh_quotient,
    power_iteration,
)
from covid.summary import mean_and_ci

import model_spec

DTYPE = model_spec.DTYPE

GIS_TEMPLATE = "data/UK2019mod_pop.gpkg"


# Reproduction number calculation
def calc_R_it(param, events, init_state, covar_data, priors):
    """Calculates effective reproduction number for batches of metapopulations
    :param theta: a tensor of batched theta parameters [B] + theta.shape
    :param xi: a tensor of batched xi parameters [B] + xi.shape
    :param events: a [B, M, T, X] batched events tensor
    :param init_state: the initial state of the epidemic at earliest inference date
    :param covar_data: the covariate data
    :return a batched vector of R_it estimates
    """

    def r_fn(args):
        beta1_, beta2_, beta3_, sigma_, xi_, gamma0_, events_ = args
        t = events_.shape[-2] - 1
        state = compute_state(init_state, events_, model_spec.STOICHIOMETRY)
        state = tf.gather(state, t, axis=-2)  # State on final inference day

        model = model_spec.CovidUK(
            covariates=covar_data,
            initial_state=init_state,
            initial_step=0,
            num_steps=events_.shape[-2],
            priors=priors,
        )

        xi_pred = model_spec.conditional_gp(
            model.model["xi"](beta1_, sigma_),
            xi_,
            tf.constant(
                [events.shape[-2] + model_spec.XI_FREQ], dtype=model_spec.DTYPE
            )[:, tf.newaxis],
        )

        par = dict(
            beta1=beta1_,
            beta2=beta2_,
            beta3=beta3_,
            sigma=sigma_,
            gamma0=gamma0_,
            xi=xi_,
        )
        print("xi shape:", par["xi"].shape)
        ngm_fn = model_spec.next_generation_matrix_fn(covar_data, par)
        ngm = ngm_fn(t, state)
        return ngm

    return tf.vectorized_map(
        r_fn,
        elems=(
            param["beta1"],
            param["beta2"],
            param["beta3"],
            param["sigma"],
            param["xi"],
            param["gamma0"],
            events,
        ),
    )


@tf.function
def predicted_incidence(param, init_state, init_step, num_steps, priors):
    """Runs the simulation forward in time from `init_state` at time `init_time`
       for `num_steps`.
    :param theta: a tensor of batched theta parameters [B] + theta.shape
    :param xi: a tensor of batched xi parameters [B] + xi.shape
    :param events: a [B, M, S] batched state tensor
    :param init_step: the initial time step
    :param num_steps: the number of steps to simulate
    :param priors: the priors for gamma
    :returns: a tensor of srt_quhape [B, M, num_steps, X] where X is the number of state
              transitions
    """

    def sim_fn(args):
        beta1_, beta2_, beta3_, sigma_, xi_, gamma0_, gamma1_, init_ = args

        par = dict(
            beta1=beta1_,
            beta2=beta2_,
            beta3=beta3_,
            gamma0=gamma0_,
            gamma1=gamma1_,
            xi=xi_,
        )

        model = model_spec.CovidUK(
            covar_data,
            initial_state=init_,
            initial_step=init_step,
            num_steps=num_steps,
            priors=priors,
        )
        sim = model.sample(**par)
        return sim["seir"]

    events = tf.map_fn(
        sim_fn,
        elems=(
            param["beta1"],
            param["beta2"],
            param["beta3"],
            param["sigma"],
            param["xi"],
            param["gamma0"],
            param["gamma1"],
            init_state,
        ),
        fn_output_signature=(tf.float64),
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
        config["data"],
        date_low=inference_period[0],
        date_high=inference_period[1],
    )

    # Load posterior file
    posterior_path = os.path.join(
        config["output"]["results_dir"], config["output"]["posterior"]
    )
    print("Using posterior:", posterior_path)
    posterior = h5py.File(
        os.path.expandvars(
            posterior_path,
        ),
        "r",
        rdcc_nbytes=1024 ** 3,
        rdcc_nslots=1e6,
    )

    # Pre-determined thinning of posterior (better done in MCMC?)
    idx = range(6000, 10000, 10)
    param = dict(
        beta1=posterior["samples/beta1"][idx],
        beta2=posterior["samples/beta2"][idx],
        beta3=posterior["samples/beta3"][
            idx,
        ],
        sigma=posterior["samples/sigma"][
            idx,
        ],
        xi=posterior["samples/xi"][idx],
        gamma0=posterior["samples/gamma0"][idx],
        gamma1=posterior["samples/gamma1"][idx],
    )
    events = posterior["samples/events"][idx]
    init_state = posterior["initial_state"][:]
    state_timeseries = compute_state(
        init_state, events, model_spec.STOICHIOMETRY
    )

    # Build model
    model = model_spec.CovidUK(
        covar_data,
        initial_state=init_state,
        initial_step=0,
        num_steps=events.shape[1],
        priors=config["mcmc"]["prior"],
    )

    ngms = calc_R_it(
        param, events, init_state, covar_data, config["mcmc"]["prior"]
    )
    b, _ = power_iteration(ngms)
    rt = rayleigh_quotient(ngms, b)
    q = np.arange(0.05, 1.0, 0.05)
    rt_quantiles = pd.DataFrame(
        {"Rt": np.quantile(rt, q, axis=-1)}, index=q
    ).T.to_excel(
        os.path.join(
            config["output"]["results_dir"], config["output"]["national_rt"]
        ),
    )

    # Prediction requires simulation from the last available timepoint for 28 + 4 + 1 days
    # Note a 4 day recording lag in the case timeseries data requires that
    # now = state_timeseries.shape[-2] + 4
    prediction = predicted_incidence(
        param,
        init_state=state_timeseries[..., -1, :],
        init_step=state_timeseries.shape[-2] - 1,
        num_steps=70,
        priors=config["mcmc"]["prior"],
    )
    predicted_state = compute_state(
        state_timeseries[..., -1, :], prediction, model_spec.STOICHIOMETRY
    )

    # Prevalence now
    prev_now = prevalence(
        predicted_state[..., 4, :], covar_data["N"], name="prev"
    )

    # Incidence of detections now
    cases_now = predicted_events(prediction[..., 4:5, 2], name="cases")

    # Incidence from now to now+7
    cases_7 = predicted_events(prediction[..., 4:11, 2], name="cases7")
    cases_14 = predicted_events(prediction[..., 4:18, 2], name="cases14")
    cases_21 = predicted_events(prediction[..., 4:25, 2], name="cases21")
    cases_28 = predicted_events(prediction[..., 4:32, 2], name="cases28")
    cases_56 = predicted_events(prediction[..., 4:60, 2], name="cases56")

    # Prevalence at day 7
    prev_7 = prevalence(
        predicted_state[..., 11, :], covar_data["N"], name="prev7"
    )
    prev_14 = prevalence(
        predicted_state[..., 18, :], covar_data["N"], name="prev14"
    )
    prev_21 = prevalence(
        predicted_state[..., 25, :], covar_data["N"], name="prev21"
    )
    prev_28 = prevalence(
        predicted_state[..., 32, :], covar_data["N"], name="prev28"
    )
    prev_56 = prevalence(
        predicted_state[..., 60, :], covar_data["N"], name="prev56"
    )

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
    rti = tf.reduce_sum(ngms, axis=-2)

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
            prev_56,
            cases_7,
            cases_14,
            cases_21,
            cases_28,
            cases_56,
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
        os.path.join(
            config["output"]["results_dir"], config["output"]["geopackage"]
        ),
        driver="GPKG",
    )
