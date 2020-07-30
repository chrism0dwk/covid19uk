"""MCMC Update classes for stochastic epidemic models"""
import numpy as np
from collections import namedtuple
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
from tensorflow_probability.python.mcmc.random_walk_metropolis import (
    UncalibratedRandomWalkResults,
)

import covid.config

DTYPE = covid.config.floatX

tfd = tfp.distributions


def random_walk_mvnorm_fn(covariance, p_u=0.95, name=None):
    """Returns callable that adds Multivariate Normal noise to the input
    :param covariance: the covariance matrix of the mvnorm proposal
    :param p_u: the bounded convergence parameter.  If equal to 1, then all proposals are drawn from the
                MVN(0, covariance) distribution, if less than 1, proposals are drawn from MVN(0, covariance)
                with probabilit p_u, and MVN(0, 0.1^2I_d/d) otherwise.
    :returns: a function implementing the proposal.
    """
    covariance = covariance + tf.eye(covariance.shape[0], dtype=DTYPE) * 1.0e-9
    scale_tril = tf.linalg.cholesky(covariance)
    rv_adapt = tfp.distributions.MultivariateNormalTriL(
        loc=tf.zeros(covariance.shape[0], dtype=DTYPE), scale_tril=scale_tril
    )
    rv_fix = tfp.distributions.Normal(
        loc=tf.zeros(covariance.shape[0], dtype=DTYPE),
        scale=0.01 / covariance.shape[0],
    )
    u = tfp.distributions.Bernoulli(probs=p_u)

    def _fn(state_parts, seed):
        with tf.name_scope(name or "random_walk_mvnorm_fn"):

            def proposal():
                rv = tf.stack([rv_fix.sample(), rv_adapt.sample()])
                uv = u.sample(seed=seed)
                return tf.gather(rv, uv)

            new_state_parts = [proposal() + state_part for state_part in state_parts]
            return new_state_parts

    return _fn


class UncalibratedLogRandomWalk(tfp.mcmc.UncalibratedRandomWalk):
    def one_step(self, current_state, previous_kernel_results):
        with tf.name_scope(mcmc_util.make_name(self.name, "rwm", "one_step")):
            with tf.name_scope("initialize"):
                if mcmc_util.is_list_like(current_state):
                    current_state_parts = list(current_state)
                else:
                    current_state_parts = [current_state]
                current_state_parts = [
                    tf.convert_to_tensor(s, name="current_state")
                    for s in current_state_parts
                ]

            # Log random walk
            next_state_parts = self.new_state_fn(
                [tf.zeros_like(s) for s in current_state_parts],
                # pylint: disable=not-callable
                self._seed_stream(),
            )
            next_state_parts = [
                cs * tf.exp(ns) for cs, ns in zip(current_state_parts, next_state_parts)
            ]

            # Compute `target_log_prob` so its available to MetropolisHastings.
            next_target_log_prob = self.target_log_prob_fn(
                *next_state_parts
            )  # pylint: disable=not-callable

            def maybe_flatten(x):
                return x if mcmc_util.is_list_like(current_state) else x[0]

            log_acceptance_correction = tf.reduce_sum(next_state_parts) - tf.reduce_sum(
                current_state_parts
            )

            return [
                maybe_flatten(next_state_parts),
                UncalibratedRandomWalkResults(
                    log_acceptance_correction=log_acceptance_correction,
                    target_log_prob=next_target_log_prob,
                ),
            ]


def get_accepted_results(results):
    if hasattr(results, "accepted_results"):
        return results.accepted_results
    else:
        return get_accepted_results(results.inner_results)


def set_accepted_results(results, accepted_results):
    if hasattr(results, "accepted_results"):
        results = results._replace(accepted_results=accepted_results)
        return results
    else:
        next_inner_results = set_accepted_results(
            results.inner_results, accepted_results
        )
        return results._replace(inner_results=next_inner_results)


def advance_target_log_prob(next_results, previous_results):
    prev_accepted_results = get_accepted_results(previous_results)
    next_accepted_results = get_accepted_results(next_results)
    next_accepted_results = next_accepted_results._replace(
        target_log_prob=prev_accepted_results.target_log_prob
    )
    return set_accepted_results(next_results, next_accepted_results)


class MH_within_Gibbs(tfp.mcmc.TransitionKernel):
    def __init__(self, target_log_prob_fn, make_kernel_fns):
        """Metropolis within Gibbs sampling.

        Based on Gibbs idea posted by Pavel Sountsov https://github.com/tensorflow/probability/issues/495

        :param target_log_prob_fn: a function which given a list of state parts calculated the joint logp
        :param make_kernel_fns: a list of functions that return an MH-compatible kernel.  Functions accept a
                                log_prob function which in turn takes a state part.
        """
        self._target_log_prob_fn = target_log_prob_fn
        self._make_kernel_fns = make_kernel_fns

    def is_calibrated(self):
        return True

    def one_step(self, state, step_results):
        prev_step = np.roll(np.arange(len(self._make_kernel_fns)), 1)
        for i, make_kernel_fn in enumerate(self._make_kernel_fns):

            def _target_log_prob_fn_part(state_part):
                state[i] = state_part
                return self._target_log_prob_fn(*state)

            kernel = make_kernel_fn(_target_log_prob_fn_part)
            results = advance_target_log_prob(
                step_results[i], step_results[prev_step[i]]
            ) or kernel.bootstrap_results(state[i])
            state[i], step_results[i] = kernel.one_step(state[i], results)
        return state, step_results

    def bootstrap_results(self, state):
        results = []
        for i, make_kernel_fn in enumerate(self._make_kernel_fns):

            def _target_log_prob_fn_part(state_part):
                state[i] = state_part
                return self._target_log_prob_fn(*state)

            kernel = make_kernel_fn(_target_log_prob_fn_part)
            results.append(kernel.bootstrap_results(init_state=state[i]))

        return results
