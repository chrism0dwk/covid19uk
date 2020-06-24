"""Mechanism for proposing event times to move"""
import collections
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.distributions.categorical import (
    _broadcast_cat_event_and_params,
)
from covid.impl.UniformInteger import UniformInteger
from covid.impl.KCategorical import KCategorical

tfd = tfp.distributions

TransitionTopology = collections.namedtuple(
    "TransitionTopology", ("prev", "target", "next")
)


def _events_or_inf(events, transition_id):
    if transition_id is None:
        return tf.fill(events.shape[:-1], tf.constant(np.inf, dtype=events.dtype))
    return tf.gather(events, transition_id, axis=-1)


def _abscumdiff(
    events, initial_state, topology, t, delta_t, bound_times, int_dtype=tf.int32
):
    """Returns the number of free events to move in target_events
       bounded by max([N_{target_id}(t)-N_{bound_id}(t)]_{bound_t}).
    :param events: a [(M), T, X] tensor of transition events
    :param initial_state: a [M, X] tensor of the constraining initial state
    :param target_id: the Xth index of the target event
    :param bound_t: the times to compute the constraints
    :param bound_id: the Xth index of the bounding event, -1 implies no bound
    :returns: a tensor of shape [M] + bound_t.shape[0] +  of max free events,
              dtype=target_events.dtype
    """

    # This line prevents negative indices.  However, we must have
    # a contract that the output of the algorithm is invalid!
    bound_times = tf.clip_by_value(
        bound_times, clip_value_min=0, clip_value_max=events.shape[-2] - 1
    )

    # Maybe replace with pad to avoid unstack/stack
    prev_events = _events_or_inf(events, topology.prev)
    target_events = tf.gather(events, topology.target, axis=-1)
    next_events = _events_or_inf(events, topology.next)
    event_tensor = tf.stack([prev_events, target_events, next_events], axis=-1)

    # Compute the absolute cumulative difference between event times
    diff = event_tensor[..., 1:] - event_tensor[..., :-1]  # [m, T, 2]
    cumdiff = tf.abs(tf.cumsum(diff, axis=-2))  # cumsum along time axis

    # Create indices into cumdiff [m, d_max, 2].  Last dimension selects
    # the bound for either the previous or next event.
    indices = tf.stack(
        [
            tf.repeat(
                tf.range(events.shape[0], dtype=int_dtype), [bound_times.shape[1]]
            ),
            tf.reshape(bound_times, [-1]),
            tf.repeat(tf.where(delta_t < 0, 0, 1), [bound_times.shape[1]]),
        ],
        axis=-1,
    )
    indices = tf.reshape(indices, [events.shape[-3], bound_times.shape[1], 3])
    free_events = tf.gather_nd(cumdiff, indices)

    # Add on initial state
    indices = tf.stack(
        [
            tf.range(events.shape[0]),
            tf.where(delta_t[:, 0] < 0, topology.target, topology.target + 1),
        ],
        axis=-1,
    )
    bound_init_state = tf.gather_nd(initial_state, indices)
    free_events += bound_init_state[..., tf.newaxis]

    return free_events


class Deterministic2(tfd.Deterministic):
    def __init__(
        self,
        loc,
        atol=None,
        rtol=None,
        validate_args=False,
        allow_nan_stats=True,
        log_prob_dtype=tf.float32,
        name="Deterministic",
    ):
        parameters = dict(locals())
        super(Deterministic2, self).__init__(
            loc,
            atol=atol,
            rtol=rtol,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name,
        )
        self.log_prob_dtype = log_prob_dtype

    def _prob(self, x):
        return tf.constant(1, dtype=self.log_prob_dtype)


class Categorical2(tfd.Categorical):
    """Done to override the faulty log_prob in tfd.Categorical due to
       https://github.com/tensorflow/tensorflow/issues/40606"""

    def _log_prob(self, k):
        logits = self.logits_parameter()
        if self.validate_args:
            k = distribution_util.embed_check_integer_casting_closed(
                k, target_dtype=self.dtype
            )
        k, logits = _broadcast_cat_event_and_params(
            k, logits, base_dtype=dtype_util.base_dtype(self.dtype)
        )
        logits_normalised = tf.math.log(tf.math.softmax(logits))
        return tf.gather(logits_normalised, k, batch_dims=1)


def EventTimeProposal(
    events, initial_state, topology, d_max, n_max, dtype=tf.int32, name=None
):
    """Draws an event time move proposal.
    :param events: a [M, T, K] tensor of event times (M number of metapopulations,
                  T number of timepoints, K number of transitions)
    :param initial_state: a [M, S] tensor of initial metapopulation x state counts
    :param topology: a 3-element tuple of (previous_transition, target_transition,
                                           next_transition), eg "(s->e, e->i, i->r)"
                                           (assuming we are interested presently in e->i,
                                            `None` for boundaries)
    :param d_max: the maximum distance over which to move (in time)
    :param n_max: the maximum number of events to move
    """
    target_events = tf.gather(events, topology.target, axis=-1)
    time_interval = tf.range(d_max, dtype=dtype)

    # def t_():
    #     x = tf.cast(target_events > 0, dtype=tf.float32)
    #     logits = tf.math.log(x)
    #     # return tfd.Multinomial(total_count=1, logits=logits, name="t_")
    #     # print("logits dtype:", logits.dtype)
    #     return tfd.OneHotCategorical(logits=logits, name="t_")

    def delta_t():
        outcomes = tf.concat([tf.range(-d_max, 0), tf.range(1, d_max + 1)], axis=0)
        logits = tf.ones([events.shape[-3]] + outcomes.shape, dtype=tf.float64)
        return tfd.FiniteDiscrete(outcomes=outcomes, logits=logits, name="delta_t")

    def t():
        # Waiting for fixed tf.nn.sparse_softmax_cross_entropy_with_logits
        x = tf.cast(target_events > 0, dtype=tf.float64)  # [M, T]
        return Categorical2(logits=tf.math.log(x), name="event_coords")
        # return Deterministic2(
        #     tf.argmax(t_, axis=-1, output_type=dtype),
        #     log_prob_dtype=events.dtype,
        #     name="t",
        # )

    def x_star(t, delta_t):
        # Compute bounds
        # The limitations of XLA mean that we must calculate bounds for
        # intervals [t, t+delta_t) if delta_t > 0, and [t+delta_t, t) if
        # delta_t is < 0.
        t = t[..., tf.newaxis]
        delta_t = delta_t[..., tf.newaxis]
        bound_times = tf.where(
            delta_t < 0, t - time_interval - 1, t + time_interval  # [t+delta_t, t)
        )  # [t, t+delta_t)
        free_events = _abscumdiff(
            events=events,
            initial_state=initial_state,
            topology=topology,
            t=t,
            delta_t=delta_t,
            bound_times=bound_times,
            int_dtype=dtype,
        )

        # Mask out bits of the interval we don't need for our delta_t
        inf_mask = tf.cumsum(
            tf.one_hot(
                tf.math.abs(delta_t[:, 0]),
                d_max,
                on_value=tf.constant(np.inf, events.dtype),
                dtype=events.dtype,
            )
        )
        free_events = tf.maximum(inf_mask, free_events)
        free_events = tf.reduce_min(free_events, axis=-1)
        indices = tf.stack([tf.range(events.shape[0], dtype=dtype), t[:, 0]], axis=-1)
        available_events = tf.gather_nd(target_events, indices)
        max_events = tf.minimum(free_events, available_events)
        max_events = tf.clip_by_value(
            max_events, clip_value_min=0, clip_value_max=n_max
        )
        # Draw x_star
        return UniformInteger(low=0, high=max_events + 1)

    return tfd.JointDistributionNamed(
        dict(t=t, delta_t=delta_t, x_star=x_star), name=name
    )


def FilteredEventTimeProposal(  # pylint: disable-invalid-name
    events, initial_state, topology, m_max, d_max, n_max, dtype=tf.int32, name=None,
):
    """FilteredEventTimeProposal allows us to choose a subset of indices
    in `range(events.shape[0])` for which to propose an update.  The
    results are then broadcast back to `events.shape[0]`.

    :param events: a [M, T, X] event tensor
    :param initial_state: a [M, S] initial state tensor
    :param topology: a TransitionTopology named tuple describing the ordering
                     of events
    :param m: the number of metapopulations to move
    :param d_max: maximum distance in time to move
    :param n_max: maximum number of events to move (user defined)
    :return: an instance of a JointDistributionNamed
    """
    target_events = tf.gather(events, topology.target, axis=-1)

    def m():
        hot_meta = tf.math.count_nonzero(target_events, axis=1, keepdims=True) > 0
        hot_meta = tf.cast(tf.transpose(hot_meta), dtype=events.dtype)
        probs = hot_meta / tf.reduce_sum(hot_meta, axis=-1)
        X = KCategorical(m_max, probs, name="m")
        return X

    def move(m):
        """We select out meta-population `m` from the first
        dimension of `events`.
        :param m: a 1-D tensor of indices of meta-populations
        :return: a random variable of type `EventTimeProposal`
        """
        select_meta = tf.gather(events, m, axis=0)
        select_init = tf.gather(initial_state, m, axis=0)
        return EventTimeProposal(
            select_meta, select_init, topology, d_max, n_max, dtype=dtype, name=None,
        )

    return tfd.JointDistributionNamed(dict(m=m, move=move))
