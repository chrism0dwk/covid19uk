"""Creates a medium term prediction"""

import pickle as pkl
import numpy as np
import pandas as pd
import tensorflow as tf

from gemlib.util import compute_state
from covid import model_spec


def make_within_rate_fns(covariates, beta2):

    C = tf.convert_to_tensor(covariates["C"], dtype=model_spec.DTYPE)
    C = tf.linalg.set_diag(C, tf.zeros(C.shape[0], dtype=model_spec.DTYPE))

    W = tf.convert_to_tensor(
        tf.squeeze(covariates["W"]), dtype=model_spec.DTYPE
    )
    N = tf.convert_to_tensor(
        tf.squeeze(covariates["N"]), dtype=model_spec.DTYPE
    )

    def within_fn(t, state):
        w_idx = tf.clip_by_value(tf.cast(t, tf.int64), 0, W.shape[0] - 1)
        commute_volume = tf.gather(W, w_idx)
        rate = state[..., 2] - beta2 * state[
            ..., 2
        ] / N * commute_volume * tf.reduce_sum(C, axis=-2)
        return rate

    def between_fn(t, state):
        w_idx = tf.clip_by_value(tf.cast(t, tf.int64), 0, W.shape[0] - 1)
        commute_volume = tf.gather(W, w_idx)
        rate = (
            beta2
            * commute_volume
            * tf.linalg.matvec(C + tf.transpose(C), state[..., 2] / N)
        )
        return rate

    return within_fn, between_fn


# @tf.function
def calc_pressure_components(covariates, beta2, state):
    def atomic_fn(args):
        beta2_, state_ = args
        within_fn, between_fn = make_within_rate_fns(covariates, beta2_)
        within = within_fn(covariates["W"].shape[0], state_)
        between = between_fn(covariates["W"].shape[0], state_)
        total = within + between
        return within / total, between / total

    return tf.vectorized_map(atomic_fn, elems=(beta2, state))


def within_between(input_files, output_file):
    """Calculates PAF for within- and between-location infection.

    :param input_files: a list of [data pickle, posterior samples pickle]
    :param output_file: a csv with within/between summary
    """

    with open(input_files[0], "rb") as f:
        covar_data = pkl.load(f)

    with open(input_files[1], "rb") as f:
        samples = pkl.load(f)

    beta2 = samples["beta2"]
    events = samples["seir"]
    init_state = samples["init_state"]
    state_timeseries = compute_state(
        init_state, events, model_spec.STOICHIOMETRY
    )

    within, between = calc_pressure_components(
        covar_data, beta2, state_timeseries[..., -1, :]
    )

    df = pd.DataFrame(
        dict(
            within_mean=np.mean(within, axis=0),
            between_mean=np.mean(between, axis=0),
            p_within_gt_between=np.mean(within > between),
        ),
        index=pd.Index(covar_data["locations"]["lad19cd"], name="location"),
    )
    df.to_csv(output_file)


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "-d", "--datafile", type=str, help="Data pickle file", requied=True
    )
    parser.add_argument(
        "-s",
        "--samples",
        type=str,
        help="Posterior samples pickle",
        required=True,
    )
    parser.add_argument("-o", "--output", type=str, help="Output csv")
    args = parser.parse_args()

    within_between([args.datafile, args.samples], args.output)
