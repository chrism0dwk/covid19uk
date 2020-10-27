"""Creates a medium term prediction"""

import os
import yaml
import numpy as np
import h5py
import geopandas as gp

import tensorflow as tf

from covid.cli_arg_parse import cli_args
from covid.impl.util import compute_state
import model_spec


GIS_TEMPLATE = "data/UK2019_mod_pop.gpkg"


def make_within_rate_fns(covariates, theta, xi):

    C = tf.convert_to_tensor(covariates["C"], dtype=model_spec.DTYPE)
    C = tf.linalg.set_diag(
        C + tf.transpose(C), tf.zeros(C.shape[0], dtype=model_spec.DTYPE)
    )
    W = tf.constant(np.squeeze(covariates["W"], dtype=model_spec.DTYPE))
    N = tf.constant(np.squeeze(covariates["N"], dtype=model_spec.DTYPE))

    beta1 = theta[0]
    beta2 = theta[1]
    gamma = theta[2]

    def within_fn(t, state):
        beta = np.exp(xi)
        rate = beta * state[..., 2] / N
        return rate

    def between_fn(t, state):
        w_idx = tf.clip_by_value(tf.cast(t, tf.int64), 0, W.shape[0] - 1)
        commute_volume = tf.gather(W, w_idx)
        beta = np.exp(xi)
        rate = (
            beta
            * beta2
            * commute_volume
            * tf.linalg.matvec(C, state[..., 2] / N)
        )
        return rate

    return within_fn, between_fn


@tf.function
def calc_pressure_components(covariates, theta, xi, state):
    def atomic_fn(theta_, xi_):
        within_fn, between_fn = make_within_rate_fns(covariates, theta_, xi_)
        within = within_fn(covariates["W"].shape[0], state)
        between = between_fn(covariates["W"].shape[0], state)
        total = within + between
        return within / total, between / total

    return tf.vectorize_map(atomic_fn, elems=(theta, xi))


args = cli_args()

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
        os.path.join(
            config["output"]["results_dir"], config["output"]["posterior"]
        )
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

within, between = calc_pressure_components(
    covar_data, theta, xi[:, -1], state_timeseries[..., -1, :]
)

gpkg = gp.read_file(
    os.path.join(
        config["output"]["results_dir"], config["output"]["geopackage"]
    )
)
gpkg = gpkg[gpkg["lad19cd"].str.startswith("E")]
gpkg = gpkg.sort_values("lad19cd")

gpkg["within_mean"] = np.mean(within, axis=0)
gpkg["between_mean"] = np.mean(between, axis=0)
gpkg["p_within_gt_between"] = np.mean(within > between)

gpkg.to_file(
    os.path.join(
        config["output"]["results_dir"], config["output"]["geopackage"]
    ),
    driver="GPKG",
)
