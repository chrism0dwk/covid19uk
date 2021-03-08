"""MCMC Test Rig for COVID-19 UK model"""
# pylint: disable=E402

import pickle as pkl
from time import perf_counter

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
from covid.tasks.mcmc_kernel_factory import make_hmc_base_kernel
from covid.tasks.mcmc_kernel_factory import make_hmc_fast_adapt_kernel
from covid.tasks.mcmc_kernel_factory import make_hmc_slow_adapt_kernel
from covid.tasks.mcmc_kernel_factory import make_event_multiscan_gibbs_step

import covid.model_spec as model_spec

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


@tf.function
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


@tf.function
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


@tf.function  # (experimental_compile=True)
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

    results_dict["hmc"] = {
        "is_accepted": unnest.get_innermost(root_results[0], "is_accepted"),
        "target_log_prob": unnest.get_innermost(
            root_results[0], "target_log_prob"
        ),
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
    return {
        "beta2": draws[0][:, 0],
        "gamma0": draws[0][:, 1],
        "gamma1": draws[0][:, 2],
        "sigma": draws[0][:, 3],
        "beta3": tf.zeros([1, 5], dtype=DTYPE),
        "beta1": draws[0][:, 4],
        "xi": draws[0][:, 5:],
        "events": draws[1],
    }


def run_mcmc(
    joint_log_prob_fn,
    current_state,
    param_bijector,
    initial_conditions,
    config,
    output_file,
):

    first_window_size, slow_window_size, last_window_size = _get_window_sizes(
        config["num_adaptation_iterations"]
    )
    num_slow_windows = 4

    hmc_kernel_kwargs = {
        "step_size": 0.00001,
        "num_leapfrog_steps": 4,
        "momentum_distribution": None,
    }
    dual_averaging_kwargs = {
        "num_adaptation_steps": first_window_size,
        "target_accept_prob": 0.75,
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
        num_samples=config["num_adaptation_iterations"]
        + config["num_burst_samples"] * config["num_bursts"],
    )
    offset = 0

    # Fast adaptation sampling
    print(f"Fast window {first_window_size}")
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
        draws_to_dict(draws), first_dim_offset=offset,
    )
    posterior.write_results(trace, first_dim_offset=offset)
    offset += first_window_size

    # Slow adaptation sampling
    hmc_kernel_kwargs["step_size"] = step_size
    for slow_window_idx in range(num_slow_windows):
        window_num_draws = slow_window_size * (2 ** slow_window_idx)
        print(f"Slow window {window_num_draws}")
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
            draws_to_dict(draws), first_dim_offset=offset,
        )
        posterior.write_results(trace, first_dim_offset=offset)
        offset += window_num_draws

    # Fast adaptation sampling
    print(f"Fast window {last_window_size}")
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
        draws_to_dict(draws), first_dim_offset=offset,
    )
    posterior.write_results(trace, first_dim_offset=offset)
    offset += last_window_size

    # Fixed window sampling
    print("Sampling...")
    hmc_kernel_kwargs["step_size"] = step_size
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
            draws_to_dict(draws), first_dim_offset=offset,
        )
        posterior.write_results(
            trace, first_dim_offset=offset,
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
    cases = xarray.open_dataset(data_file, group="observations")["cases"]

    # We load in cases and impute missing infections first, since this sets the
    # time epoch which we are analysing.
    # Impute censored events, return cases
    events = model_spec.impute_censored_events(cases.astype(DTYPE))

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
    events = events[:, start_time:, :]

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
                tfb.Softplus(low=dtype_util.eps(DTYPE)),
                tfb.Identity(),
            ],
            block_sizes=[1, 2, 1, model.event_shape["xi"][0] + 1],
        )
    )

    def joint_log_prob(unconstrained_params, events):
        params = param_bij.inverse(unconstrained_params)
        return model.log_prob(
            dict(
                beta2=params[0],
                gamma0=params[1],
                gamma1=params[2],
                sigma=params[3],
                beta1=params[4],
                xi=params[5:],
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
                np.array([0.6, 0.0, 0.0, 0.1], dtype=DTYPE),
                np.zeros(
                    model.model["xi"](0.0, 0.1).event_shape[-1] + 1,
                    dtype=DTYPE,
                ),
            ],
            axis=0,
        ),
        events,
    ]
    print("Initial logpi:", joint_log_prob(*current_chain_state))

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
        data=np.array(cases.coords["time"])
        .astype(str)
        .astype(h5py.string_dtype()),
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
    parser.add_argument(
        "data_file", type=str, help="Data pickle file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    mcmc(args.data_file, args.output, config["Mcmc"])
