"""Hotspot detection given a posterior"""

import os
import yaml
import pickle as pkl
import h5py
import numpy as np
import geopandas as gp

import tensorflow as tf
from gemlib.util import compute_state

from covid.cli_arg_parse import cli_args

import model_spec

DTYPE = model_spec.DTYPE


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


def quantile_observed(observed, prediction):
    predicted_sum = tf.reduce_sum(prediction, axis=-1)
    observed_sum = tf.reduce_sum(observed, axis=-1)
    q = tf.reduce_mean(
        tf.cast(predicted_sum < observed_sum, tf.float32), axis=0
    )
    return q


if __name__ == "__main__":

    args = cli_args()

    # Get general config
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    inference_period = [
        np.datetime64(x) for x in config["Global"]["inference_period"]
    ]

    # Load covariate data
    covar_data = model_spec.read_covariates(config)

    output_folder_path = config["output"]["results_dir"]
    geopackage_path = os.path.expandvars(
        os.path.join(output_folder_path, config["output"]["geopackage"])
    )

    # Load geopackage
    geo = gp.read_file(geopackage_path)

    geo = geo[geo["lad19cd"].str.startswith("E")]  # England only, for now.
    geo = geo.sort_values("lad19cd")

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
        beta3=posterior["samples/beta3"][idx],
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

    # Prediction requires simulation from 2 weeks ago
    prediction = predicted_incidence(
        param,
        init_state=state_timeseries[..., -14, :],
        init_step=state_timeseries.shape[-2] - 14,
        num_steps=56,
        priors=config["mcmc"]["prior"],
    )

    # prediction quantiles
    q_obs7 = quantile_observed(events[0, :, -7:, 2], prediction[..., 7:14, 2])
    q_obs14 = quantile_observed(events[0, :, -14:, 2], prediction[..., :14, 2])

    geo["Pr(pred<obs)_7"] = q_obs7.numpy()
    geo["Pr(pred<obs)_14"] = q_obs14.numpy()

    geo.to_file(
        os.path.join(
            config["output"]["results_dir"], config["output"]["geopackage"]
        ),
        driver="GPKG",
    )

    with open(
        os.path.expandvars(
            os.path.join(
                output_folder_path, config["output"]["posterior_predictive"]
            )
        ),
        "wb",
    ) as f:
        pkl.dump(prediction, f)
