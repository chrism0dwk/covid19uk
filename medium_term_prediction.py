"""Creates a medium term prediction"""

import os
import yaml
import numpy as np
import h5py
import pandas as pd

import tensorflow as tf
import tensorflow_probability as tfp

from covid.cli_arg_parse import cli_args
from covid.util import compute_state
import model_spec


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


if __name__ == "__main__":

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

    # Simulate from latest time, forward 56 steps
    model = model_spec.CovidUK(
        covar_data,
        initial_state=state_timeseries[..., -1, :],
        initial_step=state_timeseries.shape[-2] - 1,
        num_steps=56,
    )

    prediction = predicted_incidence(
        theta,
        xi,
        init_state=state_timeseries[..., -1, :],
        init_step=state_timeseries.shape[-2] - 1,
        num_steps=56,
    )

    # Prediction is [K, M, T, X]
    # We require quantile for each timepoint T, summing over M
    prediction = tf.reduce_sum(prediction[..., 1], axis=-2)
    q = tf.range(5.0, 100.0, 5.0)
    quantiles = tfp.stats.percentile(prediction, q, axis=0)  # q.shape + [T]

    dates = inference_period[1] + np.arange(quantiles.shape[1])
    output = pd.DataFrame(
        {
            "Group": "Lancaster",
            "Model": "StochasticSEIR",
            "ModelType": "Pillar 1+2",
            "Version": 0.2,
            "Creation Day": dates[0].day,
            "Creation Month": dates[0].month,
            "Creation Year": dates[0].year,
            "Day of Value": [d.day for d in dates],
            "Month of Value": [d.month for d in dates],
            "Year of Value": [d.year for d in dates],
            "AgeBand": "All",
            "Geography": "England",
            "ValueType": "num_positive_tests",
            "Value": quantiles[9, :],
        }
    )
    foo = pd.DataFrame(quantiles.T, columns=[f"Quantile {qq:%.2f}" for qq in q])
    output = pd.concat([output, foo], axis=-1)
    output.to_csv(
        os.path.join(config["output"]["results_dir"], config["output"]["medium_term"]),
        index=False,
    )
