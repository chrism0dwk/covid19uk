"""MCMC Test Rig for COVID-19 UK model"""
# pylint: disable=E402

import sys

import h5py
import xarray
import tqdm
import yaml
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import unnest
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.experimental.stats import sample_stats

from gemlib.util import compute_state
from gemlib.mcmc import Posterior
from gemlib.mcmc import GibbsKernel

from covid19uk.inference.mcmc_kernel_factory import make_hmc_base_kernel
from covid19uk.inference.mcmc_kernel_factory import make_hmc_fast_adapt_kernel
from covid19uk.inference.mcmc_kernel_factory import make_hmc_slow_adapt_kernel
from covid19uk.inference.mcmc_kernel_factory import (
    make_event_multiscan_gibbs_step,
)

import covid19uk.model_spec as model_spec

tfd = tfp.distributions
tfb = tfp.bijectors
DTYPE = model_spec.DTYPE


def get_weighted_running_variance(draws):
    """Initialises online variance accumulator"""

    prev_mean, prev_var = tf.nn.moments(draws[-draws.shape[0] // 2 :], axes=[0])
    num_samples = tf.cast(
        draws.shape[0] / 2,
        dtype=dtype_util.common_dtype([prev_mean, prev_var], tf.float32),
    )
    weighted_running_variance = sample_stats.RunningVariance.from_stats(
        num_samples=num_samples, mean=prev_mean, variance=prev_var
    )
    return weighted_running_variance


def _get_window_sizes(num_adaptation_steps):
    slow_window_size = num_adaptation_steps // 21
    first_window_size = 3 * slow_window_size
    last_window_size = (
        num_adaptation_steps - 15 * slow_window_size - first_window_size
    )
    return first_window_size, slow_window_size, last_window_size


@tf.function # (autograph=False, jit_compile=False)
def _fast_adapt_window(
    num_draws,
    joint_log_prob_fn,
    initial_position,
    hmc_kernel_kwargs,
    dual_averaging_kwargs,
    event_kernel_kwargs,
    trace_fn=None,
    seed=None,
):
    """
    In the fast adaptation window, we use the
    `DualAveragingStepSizeAdaptation` kernel
    to wrap an HMC kernel.

    :param num_draws: Number of MCMC draws in window
    :param joint_log_prob_fn: joint log posterior function
    :param initial_position: initial state of the Markov chain
    :param hmc_kernel_kwargs: `HamiltonianMonteCarlo` kernel keywords args
    :param dual_averaging_kwargs: `DualAveragingStepSizeAdaptation` keyword args
    :param event_kernel_kwargs: EventTimesMH and Occult kernel args
    :param trace_fn: function to trace kernel results
    :param seed: optional random seed.
    :returns: draws, kernel results, the adapted HMC step size, and variance
              accumulator
    """
    kernel_list = [
        (
            0,
            make_hmc_fast_adapt_kernel(
                hmc_kernel_kwargs=hmc_kernel_kwargs,
                dual_averaging_kwargs=dual_averaging_kwargs,
            ),
        ),
        (1, make_event_multiscan_gibbs_step(**event_kernel_kwargs)),
    ]

    kernel = GibbsKernel(
        target_log_prob_fn=joint_log_prob_fn,
        kernel_list=kernel_list,
        name="fast_adapt",
    )

    draws, trace, fkr = tfp.mcmc.sample_chain(
        num_draws,
        initial_position,
        kernel=kernel,
        return_final_kernel_results=True,
        trace_fn=trace_fn,
        seed=seed,
    )

    weighted_running_variance = get_weighted_running_variance(draws[0])
    step_size = unnest.get_outermost(fkr.inner_results[0], "step_size")
    return draws, trace, step_size, weighted_running_variance


@tf.function # (autograph=False, jit_compile=False)
def _slow_adapt_window(
    num_draws,
    joint_log_prob_fn,
    initial_position,
    initial_running_variance,
    hmc_kernel_kwargs,
    dual_averaging_kwargs,
    event_kernel_kwargs,
    trace_fn=None,
    seed=None,
):
    """In the slow adaptation phase, we adapt the HMC
    step size and mass matrix together.

    :param num_draws: number of MCMC iterations
    :param joint_log_prob_fn: the joint posterior density function
    :param initial_position: initial Markov chain state
    :param initial_running_variance: initial variance accumulator
    :param hmc_kernel_kwargs: `HamiltonianMonteCarlo` kernel kwargs
    :param dual_averaging_kwargs: `DualAveragingStepSizeAdaptation` kwargs
    :param event_kernel_kwargs: EventTimesMH and Occults kwargs
    :param trace_fn: result trace function
    :param seed: optional random seed
    :returns: draws, kernel results, adapted step size, the variance accumulator,
              and "learned" momentum distribution for the HMC.
    """
    kernel_list = [
        (
            0,
            make_hmc_slow_adapt_kernel(
                initial_running_variance,
                hmc_kernel_kwargs,
                dual_averaging_kwargs,
            ),
        ),
        (1, make_event_multiscan_gibbs_step(**event_kernel_kwargs)),
    ]

    kernel = GibbsKernel(
        target_log_prob_fn=joint_log_prob_fn,
        kernel_list=kernel_list,
        name="slow_adapt",
    )

    draws, trace, fkr = tfp.mcmc.sample_chain(
        num_draws,
        current_state=initial_position,
        kernel=kernel,
        return_final_kernel_results=True,
        trace_fn=trace_fn,
    )

    step_size = unnest.get_outermost(fkr.inner_results[0], "step_size")
    momentum_distribution = unnest.get_outermost(
        fkr.inner_results[0], "momentum_distribution"
    )

    weighted_running_variance = get_weighted_running_variance(draws[0])

    return (
        draws,
        trace,
        step_size,
        weighted_running_variance,
        momentum_distribution,
    )


@tf.function # (autograph=False, jit_compile=False)
def _fixed_window(
    num_draws,
    joint_log_prob_fn,
    initial_position,
    hmc_kernel_kwargs,
    event_kernel_kwargs,
    trace_fn=None,
    seed=None,
):
    """Fixed step size and mass matrix HMC.

    :param num_draws: number of MCMC iterations
    :param joint_log_prob_fn: joint log posterior density function
    :param initial_position: initial Markov chain state
    :param hmc_kernel_kwargs: `HamiltonianMonteCarlo` kwargs
    :param event_kernel_kwargs: Event and Occults kwargs
    :param trace_fn: results trace function
    :param seed: optional random seed
    :returns: (draws, trace, final_kernel_results)
    """
    kernel_list = [
        (0, make_hmc_base_kernel(**hmc_kernel_kwargs)),
        (1, make_event_multiscan_gibbs_step(**event_kernel_kwargs)),
    ]

    kernel = GibbsKernel(
        target_log_prob_fn=joint_log_prob_fn,
        kernel_list=kernel_list,
        name="fixed",
    )

    return tfp.mcmc.sample_chain(
        num_draws,
        current_state=initial_position,
        kernel=kernel,
        return_final_kernel_results=True,
        trace_fn=trace_fn,
        seed=seed,
    )


def trace_results_fn(_, results):
    """Packs results into a dictionary"""
    results_dict = {}
    root_results = results.inner_results

    step_size = tf.convert_to_tensor(
        unnest.get_outermost(root_results[0], "step_size")
    )

    results_dict["hmc"] = {
        "is_accepted": unnest.get_innermost(root_results[0], "is_accepted"),
        "target_log_prob": unnest.get_innermost(
            root_results[0], "target_log_prob"
        ),
        "step_size": step_size,
    }

    def get_move_results(results):
        return {
            "is_accepted": results.is_accepted,
            "target_log_prob": results.accepted_results.target_log_prob,
            "proposed_delta": tf.stack(
                [
                    results.accepted_results.m,
                    results.accepted_results.t,
                    results.accepted_results.delta_t,
                    results.accepted_results.x_star,
                ]
            ),
        }

    res1 = root_results[1].inner_results
    results_dict["move/S->E"] = get_move_results(res1[0])
    results_dict["move/E->I"] = get_move_results(res1[1])
    results_dict["occult/S->E"] = get_move_results(res1[2])
    results_dict["occult/E->I"] = get_move_results(res1[3])

    return results_dict


def draws_to_dict(draws):
    num_locs = draws[1].shape[1]
    num_times = draws[1].shape[2]
    return {
        "psi": draws[0][:, 0],
        "sigma_space": draws[0][:, 1],
        "beta_area": draws[0][:, 2],
        "gamma0": draws[0][:, 3],
        "gamma1": draws[0][:, 4],
        "alpha_0": draws[0][:, 5],
        "alpha_t": draws[0][:, 6 : (6 + num_times - 1)],
        "spatial_effect": draws[0][
            :, (6 + num_times - 1) : (6 + num_times - 1 + num_locs)
        ],
        "seir": draws[1],
    }


def run_mcmc(
    joint_log_prob_fn,
    current_state,
    param_bijector,
    initial_conditions,
    config,
    output_file,
):

    # first_window_size, slow_window_size, last_window_size = _get_window_sizes(
    #     config["num_adaptation_iterations"]
    # )

    first_window_size = 200
    last_window_size = 50
    slow_window_size = 25
    num_slow_windows = 6

    warmup_size = int(
        first_window_size
        + slow_window_size
        * ((1 - 2 ** num_slow_windows) / (1 - 2))  # sum geometric series
        + last_window_size
    )

    hmc_kernel_kwargs = {
        "step_size": 0.1,
        "num_leapfrog_steps": 16,
        "momentum_distribution": None,
        "store_parameters_in_results": True,
    }
    dual_averaging_kwargs = {
        "target_accept_prob": 0.75,
        # "decay_rate": 0.80,
    }
    event_kernel_kwargs = {
        "initial_state": initial_conditions,
        "t_range": [
            current_state[1].shape[-2] - 21,
            current_state[1].shape[-2],
        ],
        "config": config,
    }

    # Set up posterior
    print("Initialising output...", end="", flush=True, file=sys.stderr)
    draws, trace, _ = _fixed_window(
        num_draws=1,
        joint_log_prob_fn=joint_log_prob_fn,
        initial_position=current_state,
        hmc_kernel_kwargs=hmc_kernel_kwargs,
        event_kernel_kwargs=event_kernel_kwargs,
        trace_fn=trace_results_fn,
    )
    posterior = Posterior(
        output_file,
        sample_dict=draws_to_dict(draws),
        results_dict=trace,
        num_samples=warmup_size
        + config["num_burst_samples"] * config["num_bursts"],
    )
    offset = 0
    print("Done", flush=True, file=sys.stderr)

    # Fast adaptation sampling
    print(f"Fast window {first_window_size}", file=sys.stderr, flush=True)
    dual_averaging_kwargs["num_adaptation_steps"] = first_window_size
    draws, trace, step_size, running_variance = _fast_adapt_window(
        num_draws=first_window_size,
        joint_log_prob_fn=joint_log_prob_fn,
        initial_position=current_state,
        hmc_kernel_kwargs=hmc_kernel_kwargs,
        dual_averaging_kwargs=dual_averaging_kwargs,
        event_kernel_kwargs=event_kernel_kwargs,
        trace_fn=trace_results_fn,
    )
    current_state = [s[-1] for s in draws]
    draws[0] = param_bijector.inverse(draws[0])
    posterior.write_samples(
        draws_to_dict(draws),
        first_dim_offset=offset,
    )
    posterior.write_results(trace, first_dim_offset=offset)
    offset += first_window_size

    # Slow adaptation sampling
    hmc_kernel_kwargs["step_size"] = step_size
    for slow_window_idx in range(num_slow_windows):
        window_num_draws = slow_window_size * (2 ** slow_window_idx)
        dual_averaging_kwargs["num_adaptation_steps"] = window_num_draws
        print(f"Slow window {window_num_draws}", file=sys.stderr, flush=True)
        (
            draws,
            trace,
            step_size,
            running_variance,
            momentum_distribution,
        ) = _slow_adapt_window(
            num_draws=window_num_draws,
            joint_log_prob_fn=joint_log_prob_fn,
            initial_position=current_state,
            initial_running_variance=running_variance,
            hmc_kernel_kwargs=hmc_kernel_kwargs,
            dual_averaging_kwargs=dual_averaging_kwargs,
            event_kernel_kwargs=event_kernel_kwargs,
            trace_fn=trace_results_fn,
        )
        hmc_kernel_kwargs["step_size"] = step_size
        hmc_kernel_kwargs["momentum_distribution"] = momentum_distribution
        current_state = [s[-1] for s in draws]
        draws[0] = param_bijector.inverse(draws[0])
        posterior.write_samples(
            draws_to_dict(draws),
            first_dim_offset=offset,
        )
        posterior.write_results(trace, first_dim_offset=offset)
        offset += window_num_draws

    # Fast adaptation sampling
    print(f"Fast window {last_window_size}", file=sys.stderr, flush=True)
    dual_averaging_kwargs["num_adaptation_steps"] = last_window_size
    draws, trace, step_size, _ = _fast_adapt_window(
        num_draws=last_window_size,
        joint_log_prob_fn=joint_log_prob_fn,
        initial_position=current_state,
        hmc_kernel_kwargs=hmc_kernel_kwargs,
        dual_averaging_kwargs=dual_averaging_kwargs,
        event_kernel_kwargs=event_kernel_kwargs,
        trace_fn=trace_results_fn,
    )
    current_state = [s[-1] for s in draws]
    draws[0] = param_bijector.inverse(draws[0])
    posterior.write_samples(
        draws_to_dict(draws),
        first_dim_offset=offset,
    )
    posterior.write_results(trace, first_dim_offset=offset)
    offset += last_window_size

    # Fixed window sampling
    print("Sampling...", file=sys.stderr, flush=True)
    hmc_kernel_kwargs["step_size"] = tf.reduce_mean(
        trace["hmc"]["step_size"][-last_window_size // 2 :]
    )
    print("Fixed kernel kwargs:", hmc_kernel_kwargs, flush=True)
    for i in tqdm.tqdm(
        range(config["num_bursts"]),
        unit_scale=config["num_burst_samples"] * config["thin"],
    ):
        draws, trace, _ = _fixed_window(
            num_draws=config["num_burst_samples"],
            joint_log_prob_fn=joint_log_prob_fn,
            initial_position=current_state,
            hmc_kernel_kwargs=hmc_kernel_kwargs,
            event_kernel_kwargs=event_kernel_kwargs,
            trace_fn=trace_results_fn,
        )
        current_state = [state_part[-1] for state_part in draws]
        draws[0] = param_bijector.inverse(draws[0])
        posterior.write_samples(
            draws_to_dict(draws),
            first_dim_offset=offset,
        )
        posterior.write_results(
            trace,
            first_dim_offset=offset,
        )
        offset += config["num_burst_samples"]

    return posterior


def mcmc(data_file, output_file, config, use_autograph=False, use_xla=True):
    """Constructs and runs the MCMC"""

    if tf.test.gpu_device_name():
        print("Using GPU")
    else:
        print("Using CPU")

    data = xarray.open_dataset(data_file, group="constant_data")
    cases = xarray.open_dataset(data_file, group="observations")[
        "cases"
    ].astype(DTYPE)
    dates = cases.coords["time"]

    # Impute censored events, return cases
    # Take the last week of data, and repeat a further 3 times
    # to get a better occult initialisation.
    extra_cases = tf.tile(cases[:, -7:], [1, 3])
    cases = tf.concat([cases, extra_cases], axis=-1)
    events = model_spec.impute_censored_events(cases).numpy()

    # Initial conditions are calculated by calculating the state
    # at the beginning of the inference period
    #
    # Imputed censored events that pre-date the first I-R events
    # in the cases dataset are discarded.  They are only used to
    # to set up a sensible initial state.
    state = compute_state(
        initial_state=tf.concat(
            [
                tf.constant(data["N"], DTYPE)[:, tf.newaxis],
                tf.zeros_like(events[:, 0, :]),
            ],
            axis=-1,
        ),
        events=events,
        stoichiometry=model_spec.STOICHIOMETRY,
    )
    start_time = state.shape[1] - cases.shape[1]
    initial_state = state[:, start_time, :]
    events = events[:, start_time:-21, :]  # Clip off the "extra" events

    ########################################################
    # Construct the MCMC kernels #
    ########################################################
    model = model_spec.CovidUK(
        covariates=data,
        initial_state=initial_state,
        initial_step=0,
        num_steps=events.shape[1],
    )

    param_bij = tfb.Invert(  # Forward transform unconstrains params
        tfb.Blockwise(
            [
                tfb.Softplus(low=dtype_util.eps(DTYPE)),
                tfb.Identity(),
                tfb.Identity(),
                tfb.Identity(),
            ],
            block_sizes=[2, 4, events.shape[1] - 1, events.shape[0]],
        )
    )

    def joint_log_prob(unconstrained_params, events):
        params = param_bij.inverse(unconstrained_params)
        return model.log_prob(
            dict(
                psi=params[0],
                sigma_space=params[1],
                beta_area=params[2],
                gamma0=params[3],
                gamma1=params[4],
                alpha_0=params[5],
                alpha_t=params[6 : (6 + events.shape[1] - 1)],
                spatial_effect=params[
                    (6 + events.shape[1] - 1) : (
                        6 + events.shape[1] - 1 + events.shape[0]
                    )
                ],
                seir=events,
            )
        ) + param_bij.inverse_log_det_jacobian(
            unconstrained_params, event_ndims=1
        )

    # MCMC tracing functions
    ###############################
    # Construct bursted MCMC loop #
    ###############################
    current_chain_state = [
        tf.concat(
            [
                np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=DTYPE),
                np.full(
                    events.shape[1] + events.shape[0],
                    0.0,
                    dtype=DTYPE,
                ),
            ],
            axis=0,
        ),
        events,
    ]
    print("Num time steps:", events.shape[1], flush=True)
    print("alpha_t shape", model.event_shape["alpha_t"], flush=True)
    print("Initial chain state:", current_chain_state[0], flush=True)
    print("Initial logpi:", joint_log_prob(*current_chain_state), flush=True)

    # Output file
    posterior = run_mcmc(
        joint_log_prob_fn=joint_log_prob,
        current_state=current_chain_state,
        param_bijector=param_bij,
        initial_conditions=initial_state,
        config=config,
        output_file=output_file,
    )
    posterior._file.create_dataset("initial_state", data=initial_state)
    posterior._file.create_dataset(
        "time",
        data=np.array(dates).astype(str).astype(h5py.string_dtype()),
    )

    print(f"Acceptance theta: {posterior['results/hmc/is_accepted'][:].mean()}")
    print(
        f"Acceptance move S->E: {posterior['results/move/S->E/is_accepted'][:].mean()}"
    )
    print(
        f"Acceptance move E->I: {posterior['results/move/E->I/is_accepted'][:].mean()}"
    )
    print(
        f"Acceptance occult S->E: {posterior['results/occult/S->E/is_accepted'][:].mean()}"
    )
    print(
        f"Acceptance occult E->I: {posterior['results/occult/E->I/is_accepted'][:].mean()}"
    )

    del posterior


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser(description="Run MCMC inference algorithm")
    parser.add_argument(
        "-c", "--config", type=str, help="Config file", required=True
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Output file", required=True
    )
    parser.add_argument("data_file", type=str, help="Data NetCDF file")

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    mcmc(args.data_file, args.output, config["Mcmc"])
