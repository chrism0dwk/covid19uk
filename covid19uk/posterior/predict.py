"""Run predictions for COVID-19 model"""

import numpy as np
import xarray
import pickle as pkl
import pandas as pd
import tensorflow as tf

from covid19uk import model_spec
from covid19uk.util import copy_nc_attrs
from gemlib.util import compute_state


def predicted_incidence(
    posterior_samples,
    init_state,
    covar_data,
    init_step,
    num_steps,
    out_of_sample=False,
):
    """Runs the simulation forward in time from `init_state` at time `init_time`
       for `num_steps`.
    :param param: a dictionary of model parameters
    :covar_data: a dictionary of model covariate data
    :param init_step: the initial time step
    :param num_steps: the number of steps to simulate
    :returns: a tensor of srt_quhape [B, M, num_steps, X] where X is the number of state
              transitions
    """

    posterior_state = compute_state(
        init_state, posterior_samples["seir"], model_spec.STOICHIOMETRY,
    )
    posterior_samples["new_init_state"] = posterior_state[..., init_step, :]
    del posterior_samples["seir"]

    # For out-of-sample prediction, we have to re-simulate the
    # alpha_t trajectory given the starting point.
    if out_of_sample is True:
        alpha_t = posterior_samples["alpha_0"][:, tf.newaxis] + tf.cumsum(
            posterior_samples["alpha_t"], axis=-1
        )
        if init_step > 0:
            posterior_samples["alpha_0"] = alpha_t[:, init_step - 1]

        # Remove alpha_t from the posterior to make TFP re-simulate it.
        del posterior_samples["alpha_t"]

    @tf.function
    def do_sim():
        def sim_fn(args):
            par = tf.nest.pack_sequence_as(posterior_samples, args)
            init_ = par["new_init_state"]
            del par["new_init_state"]

            model = model_spec.CovidUK(
                covar_data,
                initial_state=init_,
                initial_step=init_step,
                num_steps=num_steps,
            )
            sim = model.sample(**par)
            return sim["seir"]

        return tf.map_fn(
            sim_fn,
            elems=tf.nest.flatten(posterior_samples),
            fn_output_signature=(tf.float64),
        )

    return posterior_samples["new_init_state"], do_sim()


def read_pkl(filename):
    with open(filename, "rb") as f:
        return pkl.load(f)


def predict(
    data,
    posterior_samples,
    output_file,
    initial_step,
    num_steps,
    out_of_sample=False,
):

    covar_data = xarray.open_dataset(data, group="constant_data")
    cases = xarray.open_dataset(data, group="observations")

    samples = read_pkl(posterior_samples)
    initial_state = samples["initial_state"]
    del samples["initial_state"]

    if initial_step < 0:
        initial_step = samples["seir"].shape[-2] + initial_step

    origin_date = np.array(cases.coords["time"][0])
    dates = np.arange(
        origin_date,
        origin_date + np.timedelta64(initial_step + num_steps, "D"),
        np.timedelta64(1, "D"),
    )

    covar_data["weekday"] = xarray.DataArray(
        (pd.to_datetime(dates).weekday < 5).astype(model_spec.DTYPE),
        coords=[dates],
        dims=["prediction_time"],
    )

    estimated_init_state, predicted_events = predicted_incidence(
        samples,
        initial_state,
        covar_data,
        initial_step,
        num_steps,
        out_of_sample,
    )

    prediction = xarray.DataArray(
        predicted_events.numpy(),
        coords=[
            np.arange(predicted_events.shape[0]),
            covar_data.coords["location"],
            dates[initial_step:],
            np.arange(predicted_events.shape[3]),
        ],
        dims=("iteration", "location", "time", "event"),
    )
    estimated_init_state = xarray.DataArray(
        estimated_init_state.numpy(),
        coords=[
            np.arange(estimated_init_state.shape[0]),
            covar_data.coords["location"],
            np.arange(estimated_init_state.shape[-1]),
        ],
        dims=("iteration", "location", "state"),
    )
    ds = xarray.Dataset(
        {"events": prediction, "initial_state": estimated_init_state}
    )
    ds.to_netcdf(output_file, group="predictions")
    ds.close()
    copy_nc_attrs(data, output_file)


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "-i", "--initial-step", type=int, default=0, help="Initial step"
    )
    parser.add_argument(
        "-n", "--num-steps", type=int, default=1, help="Number of steps"
    )
    parser.add_argument(
        "-o",
        "--out-of-sample",
        action="store_true",
        help="Out of sample prediction (sample alpha_t)",
    )
    parser.add_argument("data_pkl", type=str, help="Covariate data pickle")
    parser.add_argument(
        "posterior_samples_pkl", type=str, help="Posterior samples pickle",
    )
    parser.add_argument(
        "output_file", type=str, help="Output pkl file",
    )
    args = parser.parse_args()

    predict(
        args.data_pkl,
        args.posterior_samples_pkl,
        args.output_file,
        args.initial_step,
        args.num_steps,
        args.out_of_sample,
    )
