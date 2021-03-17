"""Calculates and saves a next generation matrix"""

import pickle as pkl
import numpy as np
import xarray
import tensorflow as tf

from covid import model_spec
from covid.util import copy_nc_attrs
from gemlib.util import compute_state


def calc_posterior_rit(samples, initial_state, times, covar_data):
    """Calculates effective reproduction number for batches of metapopulations
    :param theta: a tensor of batched theta parameters [B] + theta.shape
    :param xi: a tensor of batched xi parameters [B] + xi.shape
    :param events: a [B, M, T, X] batched events tensor
    :param init_state: the initial state of the epidemic at earliest inference date
    :param covar_data: the covariate data
    :return a batched vector of R_it estimates
    """
    times = tf.convert_to_tensor(times)

    def r_fn(args):

        par = tf.nest.pack_sequence_as(samples, args)

        state = compute_state(
            initial_state, par["seir"], model_spec.STOICHIOMETRY
        )
        del par["seir"]

        def fn(t):
            state_ = tf.gather(
                state, t, axis=-2
            )  # State on final inference day
            ngm_fn = model_spec.next_generation_matrix_fn(covar_data, par)
            ngm = ngm_fn(t, state_)
            return ngm

        ngm = tf.vectorized_map(fn, elems=times)
        return tf.reduce_sum(ngm, axis=-2)  # sum over destinations

    return tf.vectorized_map(
        r_fn,
        elems=tf.nest.flatten(samples),
    )


CHUNKSIZE = 50


def reproduction_number(input_files, output_file):

    covar_data = xarray.open_dataset(input_files[0], group="constant_data")

    with open(input_files[1], "rb") as f:
        samples = pkl.load(f)
    num_samples = samples["seir"].shape[0]

    initial_state = samples["initial_state"]
    del samples["initial_state"]

    times = np.arange(covar_data.coords["time"].shape[0])

    # Compute ngm posterior in chunks to prevent over-memory
    r_its = []
    for i in range(0, num_samples, CHUNKSIZE):
        start = i
        end = np.minimum(i + CHUNKSIZE, num_samples)
        print(f"Chunk {start}:{end}", flush=True)
        subsamples = {k: v[start:end] for k, v in samples.items()}
        r_it = calc_posterior_rit(subsamples, initial_state, times, covar_data)
        r_its.append(r_it)

    r_it = xarray.DataArray(
        tf.concat(r_its, axis=0),
        coords=[
            np.arange(num_samples),
            covar_data.coords["time"][times],
            covar_data.coords["location"],
        ],
        dims=["iteration", "time", "location"],
    )
    weight = covar_data["N"] / covar_data["N"].sum()
    r_t = (r_it * weight).sum(dim="location")
    ds = xarray.Dataset({"R_it": r_it, "R_t": r_t})

    # Output
    ds.to_netcdf(output_file, group="posterior_predictive")
    copy_nc_attrs(input_files[0], output_file)


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "samples",
        type=str,
        help="A pickle file with MCMC samples",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        help="A data glob pickle file",
        required=True,
    )
    parser.add_argument(
        "-o", "--output", type=str, help="The output file", required=True
    )
    args = parser.parse_args()

    reproduction_number([args.data, args.samples], args.output)
