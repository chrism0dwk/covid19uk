"""MCMC kernel builder functions"""

import tensorflow_probability as tfp

from gemlib.mcmc import UncalibratedEventTimesUpdate
from gemlib.mcmc import UncalibratedOccultUpdate
from gemlib.mcmc import TransitionTopology
from gemlib.mcmc import MultiScanKernel
from gemlib.mcmc import GibbsKernel


# Kernels
# Build Metropolis within Gibbs sampler with windowed HMC
def make_hmc_base_kernel(
    step_size,
    num_leapfrog_steps,
    momentum_distribution,
):
    def fn(target_log_prob_fn, _):
        return tfp.experimental.mcmc.PreconditionedHamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
            momentum_distribution=momentum_distribution,
        )

    return fn


def make_hmc_fast_adapt_kernel(
    hmc_kernel_kwargs,
    dual_averaging_kwargs,
):
    def fn(target_log_prob_fn, state):
        return tfp.mcmc.DualAveragingStepSizeAdaptation(
            make_hmc_base_kernel(
                **hmc_kernel_kwargs,
            )(target_log_prob_fn, state),
            **dual_averaging_kwargs,
        )

    return fn


def make_hmc_slow_adapt_kernel(
    initial_running_variance,
    hmc_kernel_kwargs,
    dual_averaging_kwargs,
):
    def fn(target_log_prob_fn, state):
        return tfp.experimental.mcmc.DiagonalMassMatrixAdaptation(
            make_hmc_fast_adapt_kernel(
                hmc_kernel_kwargs, dual_averaging_kwargs
            )(target_log_prob_fn, state),
            initial_running_variance=initial_running_variance,
        )

    return fn


def make_partially_observed_step(
    initial_state,
    target_event_id,
    prev_event_id,
    next_event_id,
    config,
    name=None,
):
    def fn(target_log_prob_fn, _):
        return tfp.mcmc.MetropolisHastings(
            inner_kernel=UncalibratedEventTimesUpdate(
                target_log_prob_fn=target_log_prob_fn,
                target_event_id=target_event_id,
                prev_event_id=prev_event_id,
                next_event_id=next_event_id,
                initial_state=initial_state,
                dmax=config["dmax"],
                mmax=config["m"],
                nmax=config["nmax"],
            ),
            name=name,
        )

    return fn


def make_occults_step(
    initial_state,
    t_range,
    prev_event_id,
    target_event_id,
    next_event_id,
    config,
    name,
):
    def fn(target_log_prob_fn, _):
        return tfp.mcmc.MetropolisHastings(
            inner_kernel=UncalibratedOccultUpdate(
                target_log_prob_fn=target_log_prob_fn,
                topology=TransitionTopology(
                    prev_event_id, target_event_id, next_event_id
                ),
                cumulative_event_offset=initial_state,
                nmax=config["occult_nmax"],
                t_range=t_range,
                name=name,
            ),
            name=name,
        )

    return fn


def make_event_multiscan_gibbs_step(
    initial_state,
    t_range,
    config,
):
    def make_kernel_fn(target_log_prob_fn, _):
        return MultiScanKernel(
            config["num_event_time_updates"],
            GibbsKernel(
                target_log_prob_fn=target_log_prob_fn,
                kernel_list=[
                    (
                        0,
                        make_partially_observed_step(
                            initial_state, 0, None, 1, config, "se_events"
                        ),
                    ),
                    (
                        0,
                        make_partially_observed_step(
                            initial_state, 1, 0, 2, config, "ei_events"
                        ),
                    ),
                    (
                        0,
                        make_occults_step(
                            initial_state,
                            t_range,
                            None,
                            0,
                            1,
                            config,
                            "se_occults",
                        ),
                    ),
                    (
                        0,
                        make_occults_step(
                            initial_state,
                            t_range,
                            0,
                            1,
                            2,
                            config,
                            "ei_occults",
                        ),
                    ),
                ],
                name="gibbs1",
            ),
        )

    return make_kernel_fn
