"""Run predictions for COVID-19 model"""

import numpy as np
import xarray
import pickle as pkl
import tensorflow as tf

from covid import model_spec
from covid.util import copy_nc_attrs
from gemlib.util import compute_state


def predicted_incidence(posterior_samples, covar_data, init_step, num_steps):
    """Runs the simulation forward in time from `init_state` at time `init_time`
       for `num_steps`.
    :param param: a dictionary of model parameters
    :covar_data: a dictionary of model covariate data
    :param init_step: the initial time step
    :param num_steps: the number of steps to simulate
    :returns: a tensor of srt_quhape [B, M, num_steps, X] where X is the number of state
              transitions
    """

    @tf.function
    def sim_fn(args):
        beta1_, beta2_, sigma_, xi_, gamma0_, gamma1_, init_ = args

        par = dict(
            beta1=beta1_,
            beta2=beta2_,
            sigma=sigma_,
            xi=xi_,
            gamma0=gamma0_,
            gamma1=gamma1_,
        )
        model = model_spec.CovidUK(
            covar_data,
            initial_state=init_,
            initial_step=init_step,
            num_steps=num_steps,
        )
        sim = model.sample(**par)
        return sim["seir"]

    posterior_state = compute_state(
        posterior_samples["init_state"],
        posterior_samples["seir"],
        model_spec.STOICHIOMETRY,
    )
    init_state = posterior_state[..., init_step, :]

    events = tf.map_fn(
        sim_fn,
        elems=(
            posterior_samples["beta1"],
            posterior_samples["beta2"],
            posterior_samples["sigma"],
            posterior_samples["xi"],
            posterior_samples["gamma0"],
            posterior_samples["gamma1"],
            init_state,
        ),
        fn_output_signature=(tf.float64),
    )
    return init_state, events


def read_pkl(filename):
    with open(filename, "rb") as f:
        return pkl.load(f)


def predict(data, posterior_samples, output_file, initial_step, num_steps):

    covar_data = xarray.open_dataset(data, group="constant_data")
    cases = xarray.open_dataset(data, group="observations")
    samples = read_pkl(posterior_samples)

    if initial_step < 0:
        initial_step = samples["seir"].shape[-2] + initial_step

    origin_date = np.array(cases.coords["time"][0])
    dates = np.arange(
        origin_date + np.timedelta64(initial_step, "D"),
        origin_date + np.timedelta64(initial_step + num_steps, "D"),
        np.timedelta64(1, "D"),
    )

    estimated_init_state, predicted_events = predicted_incidence(
        samples, covar_data, initial_step, num_steps
    )

    prediction = xarray.DataArray(
        predicted_events,
        coords=[
            np.arange(predicted_events.shape[0]),
            covar_data.coords["location"],
            dates,
            np.arange(predicted_events.shape[3]),
        ],
        dims=("iteration", "location", "time", "event"),
    )
    estimated_init_state = xarray.DataArray(
        estimated_init_state,
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
        "-i", "--initial-step", type=int, default=0, description="Initial step"
    )
    parser.add_argument(
        "-n", "--num-steps", type=int, default=1, description="Number of steps"
    )
    parser.add_argument(
        "data_pkl", type=str, description="Covariate data pickle"
    )
    parser.add_argument(
        "posterior_samples_pkl",
        type=str,
        description="Posterior samples pickle",
    )
    parser.add_argument(
        "output_file",
        type=str,
        description="Output pkl file",
    )
    args = parser.parse_args()

    predict(
        args.data_pkl,
        args.posterior_samples_pkl,
        args.output_file,
        args.initial_step,
        args.num_steps,
    )
