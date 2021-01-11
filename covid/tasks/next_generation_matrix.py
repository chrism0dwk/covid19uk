"""Calculates and saves a next generation matrix"""

import pickle as pkl
import numpy as np
import xarray
import tensorflow as tf


from covid import model_spec
from gemlib.util import compute_state


def calc_posterior_ngm(samples, covar_data):
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
        state = compute_state(
            samples["init_state"], events_, model_spec.STOICHIOMETRY
        )
        state = tf.gather(state, t, axis=-2)  # State on final inference day

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
            samples["beta1"],
            samples["beta2"],
            samples["beta3"],
            samples["sigma"],
            samples["xi"],
            samples["gamma0"],
            samples["seir"],
        ),
    )


def next_generation_matrix(input_files, output_file):

    with open(input_files[0], "rb") as f:
        covar_data = pkl.load(f)

    with open(input_files[1], "rb") as f:
        samples = pkl.load(f)

    # Compute ngm posterior
    ngm = calc_posterior_ngm(samples, covar_data).numpy()
    ngm = xarray.DataArray(
        ngm,
        coords=[
            np.arange(ngm.shape[0]),
            covar_data["locations"]["lad19cd"],
            covar_data["locations"]["lad19cd"],
        ],
        dims=["iteration", "dest", "src"],
    )
    # Output
    with open(output_file, "wb") as f:
        pkl.dump(ngm, f)


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "-s",
        "--samples",
        type=str,
        description="A pickle file with MCMC samples",
        required=True,
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        decription="A data glob pickle file",
        require=True,
    )
    parser.add_argument(
        "-o", "--output", type=str, description="The output file", require=True
    )
    args = parser.parse_args()

    next_generation_matrix([args.data, args.samples], args.output)
