"""Marginal analysis"""

import os
import numpy as np
import yaml
import h5py
import tensorflow as tf

from gemlib.distributions.state_transition_marginal_model import (
    BaselineHazardRateMarginal,
)
import model_spec_marginal as ms

DTYPE = tf.float64

BASEDIR = "/scratch/hpc/39/jewellcp/covid19/marginal_2020-10-23_both_specimen/"

with open(os.path.join(BASEDIR, "config.yaml"), "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

inference_period = [
    np.datetime64(x) for x in config["settings"]["inference_period"]
]
covariates = ms.read_covariates(config["data"], *inference_period)

post = h5py.File(
    os.path.join(BASEDIR, "posterior.hd5"), "r", rdcc_nbytes=1024 ** 3
)


def make_transition_rate_fn(beta2, xi):

    XI_FREQ = 14
    beta2 = tf.convert_to_tensor(beta2, DTYPE)
    xi = tf.convert_to_tensor(xi, DTYPE)

    def transition_rate_fn(t, state):
        C = tf.convert_to_tensor(covariates["C"], dtype=DTYPE)
        C = tf.linalg.set_diag(
            C + tf.transpose(C), tf.zeros(C.shape[0], dtype=DTYPE)
        )
        W = tf.constant(np.squeeze(covariates["W"]), dtype=DTYPE)
        N = tf.constant(np.squeeze(covariates["N"]), dtype=DTYPE)

        w_idx = tf.clip_by_value(tf.cast(t, tf.int64), 0, W.shape[0] - 1)
        commute_volume = tf.gather(W, w_idx)
        xi_idx = tf.cast(
            tf.clip_by_value(t // XI_FREQ, 0, xi.shape[0] - 1),
            dtype=tf.int64,
        )
        xi_ = tf.gather(xi, xi_idx)

        infec_rate = tf.math.exp(xi_) * (
            state[..., 2]
            + beta2
            * commute_volume
            * tf.linalg.matvec(C, state[..., 2] / tf.squeeze(N))
        )
        infec_rate = (
            infec_rate / tf.squeeze(N) + 0.000000001
        )  # Vector of length nc

        ei = tf.broadcast_to(
            [tf.constant(1.0, dtype=DTYPE)], shape=[state.shape[0]]
        )  # Vector of length nc
        ir = tf.broadcast_to(
            [tf.constant(1.0, dtype=DTYPE)], shape=[state.shape[0]]
        )  # Vector of length nc

        return [infec_rate, ei, ir]

    return transition_rate_fn


def make_baseline_hazard(beta2, xi, events):
    priors = dict(concentration=[0.1, 2.0, 2.0], rate=[0.1, 4.0, 4.0])
    bh = BaselineHazardRateMarginal(
        events,
        make_transition_rate_fn(beta2, xi),
        priors,
        ms.STOICHIOMETRY,
        post["initial_state"],
        0,
        1.0,
        events.shape[0],
    )
    return bh


def draw_baseline_hazards():
    def fn(args):
        return make_baseline_hazard(*args).sample()

    return tf.vectorized_map(
        fn=fn,
        elems=(
            post["samples/theta"][:, 0],
            post["samples/theta"][:, 1:],
            post["samples/events"][:],
        ),
    )


samples = draw_baseline_hazards()

fig, ax = plt.subplots(3)
ax[0].plot(samples[:, 0])  # beta
ax[0].set_ylabel("beta")
ax[1].plot(samples[:, 1])  # nu
ax[1].set_ylabel("nu")
ax[2].plot(samples[:, 2])  # gamma
ax[2].set_ylabel("gamma")
