"""Calculates and saves a next generation matrix"""

import argparse
import yaml
import pickle as pkl
import tensorflow as tf


from covid import model_spec
from gemlib.util import compute_state


def calc_posterior_ngm(param, events, init_state, covar_data):
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
        )

        par = dict(
            beta1=beta1_,
            beta2=beta2_,
            beta3=beta3_,
            sigma=sigma_,
            gamma0=gamma0_,
            xi=xi_,
        )
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


def next_generation_matrix(input_files, output_file):

    with open(input_files[0], "rb") as f:
        covar_data = pkl.load(f)

    with open(input_files[1], "rb") as f:
        param = pkl.load(f)

    # Compute ngm posterior
    ngm = calc_posterior_ngm(
        param, param["events"], param["init_state"], covar_data
    )

    # Output
    with open(output_file, "wb") as f:
        pkl.dump(ngm, f)


if __name__ == "__main__":

    next_generation_matrix(input_files, output_file)
