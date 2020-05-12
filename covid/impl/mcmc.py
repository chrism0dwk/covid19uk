"""MCMC Update classes for stochastic epidemic models"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
from tensorflow_probability.python.mcmc.random_walk_metropolis import UncalibratedRandomWalkResults
from tensorflow_probability.python.util import SeedStream

tfd = tfp.distributions

DTYPE = tf.float64


def random_walk_mvnorm_fn(covariance, p_u=0.95, name=None):
    """Returns callable that adds Multivariate Normal noise to the input
    :param covariance: the covariance matrix of the mvnorm proposal
    :param p_u: the bounded convergence parameter.  If equal to 1, then all proposals are drawn from the
                MVN(0, covariance) distribution, if less than 1, proposals are drawn from MVN(0, covariance)
                with probabilit p_u, and MVN(0, 0.1^2I_d/d) otherwise.
    :returns: a function implementing the proposal.
    """
    covariance = covariance + tf.eye(covariance.shape[0], dtype=DTYPE) * 1.e-9
    scale_tril = tf.linalg.cholesky(covariance)
    rv_adapt = tfp.distributions.MultivariateNormalTriL(loc=tf.zeros(covariance.shape[0], dtype=DTYPE),
                                                        scale_tril=scale_tril)
    rv_fix = tfp.distributions.Normal(loc=tf.zeros(covariance.shape[0], dtype=DTYPE),
                                      scale=0.01/covariance.shape[0],)
    u = tfp.distributions.Bernoulli(probs=p_u)

    def _fn(state_parts, seed):
        with tf.name_scope(name or 'random_walk_mvnorm_fn'):
            def proposal():
                rv = tf.stack([rv_fix.sample(), rv_adapt.sample()])
                uv = u.sample()
                return tf.gather(rv, uv)
            new_state_parts = [proposal() + state_part for state_part in state_parts]
            return new_state_parts

    return _fn


class UncalibratedLogRandomWalk(tfp.mcmc.UncalibratedRandomWalk):
    def one_step(self, current_state, previous_kernel_results):
        with tf.name_scope(mcmc_util.make_name(self.name, 'rwm', 'one_step')):
            with tf.name_scope('initialize'):
                if mcmc_util.is_list_like(current_state):
                    current_state_parts = list(current_state)
                else:
                    current_state_parts = [current_state]
                current_state_parts = [
                    tf.convert_to_tensor(s, name='current_state')
                    for s in current_state_parts
                ]

            # Log random walk
            next_state_parts = self.new_state_fn([tf.zeros_like(s) for s in current_state_parts],  # pylint: disable=not-callable
                                                 self._seed_stream())
            next_state_parts = [cs * tf.exp(ns) for cs, ns in zip(current_state_parts, next_state_parts)]

            # Compute `target_log_prob` so its available to MetropolisHastings.
            next_target_log_prob = self.target_log_prob_fn(*next_state_parts)  # pylint: disable=not-callable

            def maybe_flatten(x):
                return x if mcmc_util.is_list_like(current_state) else x[0]

            log_acceptance_correction = tf.reduce_sum(next_state_parts) - tf.reduce_sum(current_state_parts)

            return [
                maybe_flatten(next_state_parts),
                UncalibratedRandomWalkResults(
                    log_acceptance_correction=log_acceptance_correction,
                    target_log_prob=next_target_log_prob,
                ),
            ]


class DeterministicFloatX(tfd.Deterministic):
    def _prob(self, x):
        loc = tf.convert_to_tensor(self.loc)
        return tf.reduce_sum(tf.cast(tf.abs(x - loc) <= self._slack(loc), dtype=DTYPE))


def matrix_where(condition):
    nrow, ncol = condition.shape
    msk = tf.reshape(condition, [-1])
    msk_idx = tf.boolean_mask(tf.range(tf.size(msk), dtype=tf.int64), msk)
    true_coords = tf.stack([msk_idx // ncol, msk_idx % ncol],
                           axis=-1)
    return true_coords


def make_event_time_move(counts_matrix, p, alpha):
    """Returns a proposal to move infection times.

    Algorithm:
        1. Choose random batch of coordinate [day, class] in event_matrix to update;
        2. Choose a number of items to move using Binomial(n_{tm}, p)
        3. Choose a new timepoint (dim 0 in events) to move to by Uniform([-1,1]) * Poisson(alpha)

    :param counts_matrix: matrix of number of units per day (dim 0) per class (dim 1)
    :param p: probability that a unit within a day/class gets chosen to move
    :param alpha: the magnitude of the distance to move the chosen units in time.
    :returns: an instance of tfd.JointDistributionNamed over all random numbers above.
    """
    counts_matrix = tf.convert_to_tensor(counts_matrix, dtype=DTYPE)
    p = tf.convert_to_tensor(p, dtype=DTYPE)
    alpha = tf.convert_to_tensor(alpha, dtype=DTYPE)

    # Choose which elements to move
    def n_events():
        # Choose number of units at each coordinate to move
        return tfd.Binomial(counts_matrix, probs=p, name='n_events')

    def dir():
        # Sample direction to move in
        return tfd.Sample(tfd.Bernoulli(probs=tf.constant(0.5, dtype=DTYPE)), counts_matrix.shape, name='dir')

    def d_mag():
        # Sample distance to move each set of units
        return tfd.Sample(tfd.Geometric(probs=alpha, name='d_mag'), counts_matrix.shape, name='d_mag')

    def distance(dir, d_mag):
        # Compute the distance to move as product of direction and distance
        return tfd.Deterministic(
            tf.gather(tf.constant([-1, 1], dtype=tf.int64), dir) * tf.cast(d_mag, tf.int64),
            name='distance')

    return tfd.JointDistributionNamed({
        'n_events': n_events,
        'dir': dir,
        'd_mag': d_mag,
        'distance': distance})


class UncalibratedEventTimesUpdate(tfp.mcmc.TransitionKernel):
    def __init__(self,
                 target_log_prob_fn,
                 p,
                 alpha,
                 seed=None,
                 name=None):
        """An uncalibrated random walk for event times.
        :param target_log_prob_fn: the log density of the target distribution
        :param p: the proportion of events to move
        :param alpha: the magnitude of the distance over which to move
        :param seed: a random seed stream
        :param name: the name of the update step
        """
        self._target_log_prob_fn = target_log_prob_fn
        self._seed_stream = SeedStream(seed, salt='UncalibratedEventTimesUpdate')
        self._name = name
        self._parameters = dict(
            target_log_prob_fn=target_log_prob_fn,
            p=p,
            alpha=alpha,
            seed=seed,
            name=name)

    @property
    def target_log_prob_fn(self):
        return self._parameters['target_log_prob_fn']

    @property
    def transition_coord(self):
        return self._parameters['transition_coord']

    @property
    def seed(self):
        return self._parameters['seed']

    @property
    def name(self):
        return self._parameters['name']

    @property
    def parameters(self):
        """Return `dict` of ``__init__`` arguments and their values."""
        return self._parameters

    @property
    def is_calibrated(self):
        return False

    def one_step(self, current_state, previous_kernel_results):
        with tf.name_scope('uncalibrated_event_times_rw/onestep'):
            proposal = make_event_time_move(current_state,
                                            self._parameters['p'],
                                            self._parameters['alpha'])
            x_star = proposal.sample(seed=self.seed)  # This is the move to make

            # Calculate the coordinate that we'll move events to
            rows = tf.repeat(tf.range(current_state.shape[0], dtype=tf.int64), current_state.shape[1])
            rows_to = rows + tf.reshape(x_star['distance'], [-1])
            coords_to = tf.stack([rows_to, tf.tile(tf.range(current_state.shape[1], dtype=tf.int64),
                                                   [current_state.shape[0]])], axis=-1)

            # Update the state
            n_events_star = tf.scatter_nd(indices=coords_to,
                                          updates=tf.reshape(x_star['n_events'], [-1]),
                                          shape=current_state.shape)
            next_state = current_state - x_star['n_events']  # Subtract moved events
            next_state = next_state + n_events_star

            next_target_log_prob = self.target_log_prob_fn(next_state)

            log_acceptance_correction = tfd.Binomial(next_state, probs=self._parameters['p']).log_prob(
                n_events_star)
            log_acceptance_correction -= tfd.Binomial(current_state, probs=self._parameters['p']).log_prob(
                x_star['n_events'])
            log_acceptance_correction = tf.reduce_sum(log_acceptance_correction)

            # tf.print("Log q:", log_acceptance_correction)
            # tf.print("Old:", previous_kernel_results.target_log_prob)
            # tf.print("New:", next_target_log_prob)

            return [next_state,
                    tfp.mcmc.random_walk_metropolis.UncalibratedRandomWalkResults(
                        log_acceptance_correction=log_acceptance_correction,
                        target_log_prob=next_target_log_prob
                    )]

    def bootstrap_results(self, init_state):
        with tf.name_scope('uncalibrated_event_times_rw/bootstrap_results'):
            init_state = tf.convert_to_tensor(init_state, dtype=DTYPE)
            init_target_log_prob = self.target_log_prob_fn(init_state)
            return tfp.mcmc.random_walk_metropolis.UncalibratedRandomWalkResults(
                log_acceptance_correction=tf.constant(0., dtype=DTYPE),
                target_log_prob=init_target_log_prob
            )


def get_accepted_results(results):
    if hasattr(results, 'accepted_results'):
        return results.accepted_results
    else:
        return get_accepted_results(results.inner_results)


def set_accepted_results(results, accepted_results):
    if hasattr(results, 'accepted_results'):
        results  = results._replace(accepted_results=accepted_results)
        return results
    else:
        next_inner_results = set_accepted_results(results.inner_results, accepted_results)
        return results._replace(inner_results=next_inner_results)


def advance_target_log_prob(next_results, previous_results):
    prev_accepted_results = get_accepted_results(previous_results)
    next_accepted_results = get_accepted_results(next_results)
    next_accepted_results = next_accepted_results._replace(target_log_prob=prev_accepted_results.target_log_prob)
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
            results = advance_target_log_prob(step_results[i],
                                              step_results[prev_step[i]]) or kernel.bootstrap_results(state[i])
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
