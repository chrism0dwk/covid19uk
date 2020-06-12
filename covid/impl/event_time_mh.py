import collections

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.util import SeedStream

from covid import config
from covid.impl.mcmc import KernelResults
from covid.impl.util import which

tfd = tfp.distributions

DTYPE = config.floatX

TransitionTopology = collections.namedtuple('TransitionTopology',
                                            ('prev',
                                             'target',
                                             'next'))


def _is_within(x, low, high):
    """Returns true if low <= x < high"""
    return tf.logical_and(tf.less_equal(low, x), tf.less(x, high))


def _nonzero_rows(m):
    return tf.cast(tf.reduce_sum(m, axis=-1) > 0., m.dtype)


def _max_free_events(events, initial_state,
                     target_t, target_id,
                     constraint_t, constraint_id):
    """Returns the maximum number of free events to move in target_events constrained by
    constraining_events.
    :param events: a [T, M, X] tensor of transition events
    :param initial_state: a [M, X] tensor of the constraining initial state
    :param target_t: the target time
    :param target_id: the Xth index of the target event
    :param constraint_t: the Tth times of the constraint
    :param constraining_id: the Xth index of the constraining event, -1 implies no constraint
    :returns: a tensor of shape constraint_t.shape[0] + [M] of max free events, dtype=target_events.dtype
    """

    def true_fn():
        target_events_ = tf.gather(events, target_id, axis=-1)
        target_cumsum = tf.cumsum(target_events_, axis=0)
        constraining_events = tf.gather(events, constraint_id, axis=-1) # TxM
        constraining_cumsum = tf.cumsum(constraining_events, axis=0)    # TxM
        constraining_init_state = tf.gather(initial_state, constraint_id + 1, axis=-1)
        n1 = tf.gather(target_cumsum, constraint_t, axis=0)
        n2 = tf.gather(constraining_cumsum, constraint_t, axis=0)
        free_events = tf.abs(n1 - n2) + constraining_init_state
        max_free_events = tf.minimum(free_events,
                                     tf.gather(target_events_, target_t, axis=0))
        return max_free_events

    # Manual broadcasting of n_events_t is required here so that the XLA
    # compiler can guarantee that the output shapes of true_fn() and
    # false_fn() are equal.  Known shape information can thus be
    # propagated right through the algorithm, so the return value has known shape.
    def false_fn():
        n_events_t = tf.gather(events[..., target_id], target_t, axis=0)
        return tf.broadcast_to([n_events_t], [constraint_t.shape[0]] + [n_events_t.shape[0]])

    ret_val = tf.cond(constraint_id != -1, true_fn, false_fn)
    return ret_val


def _move_events(event_tensor, event_id, from_t, to_t, n_move):
    """Subtracts n_move from event_tensor[from_t, :, event_id]
    and adds n_move to event_tensor[to_t, :, event_id]."""
    num_meta = event_tensor.shape[1]
    indices = tf.stack([tf.broadcast_to(from_t, [num_meta]),  # Timepoint
                        tf.range(num_meta),  # All meta-populations
                        tf.broadcast_to([event_id], [num_meta])], axis=-1)  # Event
    # Subtract x_star from the [from_t, :, event_id] row of the state tensor
    next_state = tf.tensor_scatter_nd_sub(event_tensor, indices, n_move)
    indices = tf.stack([tf.broadcast_to(to_t, [num_meta]),
                        tf.range(num_meta),
                        tf.broadcast_to(event_id, [num_meta])], axis=-1)
    # Add x_star to the [to_t, :, event_id] row of the state tensor
    next_state = tf.tensor_scatter_nd_add(next_state, indices, n_move)
    return next_state


class EventTimesUpdate(tfp.mcmc.TransitionKernel):
    def __init__(self,
                 target_log_prob_fn,
                 target_event_id,
                 prev_event_id,
                 next_event_id,
                 initial_state,
                 dmax,
                 mmax,
                 nmax,
                 seed=None,
                 name=None):
        """A random walk Metropolis Hastings for event times.
        :param target_log_prob_fn: the log density of the target distribution
        :param target_event_id: the position in the first dimension of the events tensor that we wish to move
        :param prev_event_id: the position of the previous event in the events tensor
        :param next_event_id: the position of the next event in the events tensor
        :param initial_state: the initial state tensor
        :param dmax: maximum distance to move in time
        :param mmax: number of metapopulations to move
        :param nmax: max number of events to move
        :param seed: a random seed
        :param name: the name of the update step
        """
        self._seed_stream = SeedStream(seed, salt='EventTimesUpdate')
        self._impl = tfp.mcmc.MetropolisHastings(
            inner_kernel=UncalibratedEventTimesUpdate(target_log_prob_fn=target_log_prob_fn,
                                                      target_event_id=target_event_id,
                                                      prev_event_id=prev_event_id,
                                                      next_event_id=next_event_id,
                                                      dmax=dmax,
                                                      mmax=mmax,
                                                      nmax=nmax,
                                                      initial_state=initial_state))
        self._parameters = self._impl.inner_kernel.parameters.copy()
        self._parameters['seed'] = seed

    @property
    def target_log_prob_fn(self):
        return self._impl.inner_kernel.target_log_prob_fn

    @property
    def name(self):
        return self._impl.inner_kernel.name

    @property
    def parameters(self):
        """Return `dict` of ``__init__`` arguments and their values."""
        return self._parameters

    @property
    def is_calibrated(self):
        return True

    def one_step(self, current_state, previous_kernel_results):
        """Performs one step of an event times update.
        :param current_state: the current state tensor [TxMxX]
        :param previous_kernel_results: a named tuple of results.
        :returns: (next_state, kernel_results)
        """
        next_state, kernel_results = self._impl.one_step(current_state, previous_kernel_results)
        return next_state, kernel_results

    def bootstrap_results(self, init_state):
        kernel_results = self._impl.bootstrap_results(init_state)
        return kernel_results


class UncalibratedEventTimesUpdate(tfp.mcmc.TransitionKernel):
    def __init__(self,
                 target_log_prob_fn,
                 target_event_id,
                 prev_event_id,
                 next_event_id,
                 initial_state,
                 dmax,
                 mmax,
                 nmax,
                 seed=None,
                 name=None):
        """An uncalibrated random walk for event times.
        :param target_log_prob_fn: the log density of the target distribution
        :param target_event_id: the position in the first dimension of the events tensor that we wish to move
        :param prev_event_id: the position of the previous event in the events tensor
        :param next_event_id: the position of the next event in the events tensor
        :param initial_state: the initial state tensor
        :param seed: a random seed
        :param name: the name of the update step
        """
        self._target_log_prob_fn = target_log_prob_fn
        self._seed_stream = SeedStream(seed, salt='UncalibratedEventTimesUpdate')
        self._name = name
        self._parameters = dict(
            target_log_prob_fn=target_log_prob_fn,
            target_event_id=target_event_id,
            prev_event_id=prev_event_id,
            next_event_id=next_event_id,
            initial_state=initial_state,
            dmax=dmax,
            mmax=mmax,
            nmax=nmax,
            seed=seed,
            name=name)
        self.tx_topology = TransitionTopology(prev_event_id, target_event_id, next_event_id)
        self.time_offsets = tf.range(self.parameters['dmax'])

    @property
    def target_log_prob_fn(self):
        return self._parameters['target_log_prob_fn']

    @property
    def target_event_id(self):
        return self._parameters['target_event_id']

    @property
    def prev_event_id(self):
        return self._parameters['prev_event_id']

    @property
    def next_event_id(self):
        return self._parameters['next_event_id']

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

    def one_step(self, current_events, previous_kernel_results):
        """One update of event times.
        :param current_events: a [T, M, X] tensor containing number of events per time t, metapopulation m,
                              and transition x.
        :param previous_kernel_results: an object of type UncalibratedRandomWalkResults.
        :returns: a tuple containing new_state and UncalibratedRandomWalkResults.
        """
        with tf.name_scope('uncalibrated_event_times_rw/onestep'):
            target_events = current_events[..., self.tx_topology.target]
            num_times = target_events.shape[0]

            # 1. Choose a timepoint to move, conditional on it having events to move
            current_p = _nonzero_rows(target_events)
            current_t = tf.squeeze(tf.random.categorical(logits=[tf.math.log(current_p)],
                                                         num_samples=1,
                                                         seed=self._seed_stream(),
                                                         dtype=tf.int32))

            # 2. time_delta has a magnitude and sign -- a jump in time for which to move events
            # tfp.math.random_rademacker
            # bernoulli * 2 - 1
            u = tf.squeeze(tf.random.uniform(shape=[1], seed=self._seed_stream(),  # 0 is backwards
                                             minval=0, maxval=2, dtype=tf.int32))  # 1 is forwards
            jump_sign = tf.gather([-1, 1], u)
            jump_magnitude = tf.squeeze(tf.random.uniform([1], seed=self._seed_stream(),
                                                          minval=0, maxval=self.parameters['dmax'],
                                                          dtype=tf.int32)) + 1
            time_delta = jump_sign * jump_magnitude
            next_t = current_t + time_delta

            # Compute the constraint times (current_t, time_offsets, (target, prev, next),
            #                               events_tensor, initial state, distance)
            n_max = self.compute_constraints(current_events, current_t, time_delta)

            # Draw number to move uniformly from n_max
            p_msk = tf.cast(n_max > 0., dtype=tf.float32)
            W = tfd.OneHotCategorical(logits=tf.math.log(p_msk))
            msk = tf.cast(W.sample(), n_max.dtype)
            clip_max = 20.
            n_max = tf.clip_by_value(n_max, clip_value_min=0., clip_value_max=clip_max)
            x_star = tf.floor(tf.random.uniform(n_max.shape, minval=0., maxval=(n_max + 1.),
                                                dtype=current_events.dtype)) * msk

            # Propose next_state
            next_state = _move_events(event_tensor=current_events, event_id=self.tx_topology.target,
                                      from_t=current_t, to_t=next_t,
                                      n_move=x_star)
            next_target_log_prob = self.target_log_prob_fn(next_state)

            # Trap out-of-bounds moves that go outside [0, num_times)
            next_target_log_prob = tf.where(_is_within(next_t, 0, num_times),
                                            next_target_log_prob,
                                            tf.constant(-np.inf, dtype=current_events.dtype))

            # Calculate proposal density
            # 1. Calculate probability of choosing a timepoint
            next_p = _nonzero_rows(next_state[..., self.target_event_id])
            log_acceptance_correction = tf.math.log(tf.reduce_sum(current_p)) - \
                                        tf.math.log(tf.reduce_sum(next_p))

            # 2. Calculate probability of selecting events
            next_n_max = self.compute_constraints(next_state, next_t, -time_delta)
            next_n_max = tf.clip_by_value(next_n_max, clip_value_min=0., clip_value_max=clip_max)
            log_acceptance_correction += tf.reduce_sum(tf.math.log(n_max + 1.) - tf.math.log(next_n_max + 1.))

            # 3. Prob of choosing a non-zero element to move
            log_acceptance_correction = tf.math.log(
                tf.math.count_nonzero(n_max, dtype=log_acceptance_correction.dtype)) - tf.math.log(
                tf.math.count_nonzero(next_n_max, dtype=log_acceptance_correction.dtype))

            return [next_state,
                    KernelResults(
                        log_acceptance_correction=log_acceptance_correction,
                        target_log_prob=next_target_log_prob,
                        extra=tf.concat([x_star, n_max], axis=0)
                    )]

    def compute_constraints(self, current_events, current_t, time_delta):
        """Computes the constraints on an event time move given the move magnitude and sign
        :param current_events: an event tensor describing the state
        :param current_t: the time from which events need to be moved
        :param time_delta: the time_delta by which to move the events
        """
        constraint_time_idx = tf.where(time_delta < 0,
                                       current_t - self.time_offsets - 1,
                                       current_t + self.time_offsets)

        constraining_event_id = tf.where(time_delta < 0,
                                         self.tx_topology.prev or -1,
                                         self.tx_topology.next or -1)

        # 3. Calculate max number of events to move subject to constraints
        n_max = _max_free_events(events=current_events, initial_state=self.parameters['initial_state'],
                                 target_t=current_t, target_id=self.tx_topology.target,
                                 constraint_t=constraint_time_idx, constraint_id=constraining_event_id)
        inf_mask = tf.cumsum(tf.one_hot(tf.math.abs(time_delta),
                                        self.parameters['dmax'], dtype=tf.int32)) * tf.int32.max
        n_max = tf.reduce_min(tf.cast(inf_mask[:, None], n_max.dtype) + n_max, axis=0)
        return n_max

    def bootstrap_results(self, init_state):
        with tf.name_scope('uncalibrated_event_times_rw/bootstrap_results'):
            init_state = tf.convert_to_tensor(init_state, dtype=DTYPE)
            init_target_log_prob = self.target_log_prob_fn(init_state)
            return KernelResults(
                log_acceptance_correction=tf.constant(0., dtype=DTYPE),
                target_log_prob=init_target_log_prob,
                extra=tf.zeros(2 * init_state.shape[-2], dtype=DTYPE)
            )
