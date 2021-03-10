"""Calculates and saves a next generation matrix"""

import pickle as pkl
import numpy as np
import xarray
import tensorflow as tf


from covid import model_spec
from covid.util import copy_nc_attrs
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

        par = tf.nest.pack_sequence_as(samples, args)
        
        t = events_.shape[-2] - 1
        state = compute_state(
            samples["init_state"], par['seir'], model_spec.STOICHIOMETRY
        )
        state = tf.gather(state, t, axis=-2)  # State on final inference day

        del par['seir']
        ngm_fn = model_spec.next_generation_matrix_fn(covar_data, par)
        ngm = ngm_fn(t, state)
        return ngm

    return tf.vectorized_map(
        r_fn,
        elems=tf.nest.flatten(samples),
    )


def next_generation_matrix(input_files, output_file):

    covar_data = xarray.open_dataset(input_files[0], group="constant_data")

    with open(input_files[1], "rb") as f:
        samples = pkl.load(f)

    # Compute ngm posterior
    ngm = calc_posterior_ngm(samples, covar_data).numpy()
    ngm = xarray.DataArray(
        ngm,
        coords=[
            np.arange(ngm.shape[0]),
            covar_data.coords["location"],
            covar_data.coords["location"],
        ],
        dims=["iteration", "dest", "src"],
    )
    ngm = xarray.Dataset({"ngm": ngm})

    # Output
    ngm.to_netcdf(output_file, group="posterior_predictive")
    copy_nc_attrs(input_files[0], output_file)


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
