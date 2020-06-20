"""Mechanism for proposing event times to move"""
import collections

import tensorflow as tf
import tensorflow_probability as tfp

from covid.impl.UniformInteger import UniformInteger

tfd = tfp.distributions

TransitionTopology = collections.namedtuple('TransitionTopology',
                                            ('prev',
                                             'target',
                                             'next'))


def _abscumdiff(events, initial_state, target_id, bound_t, bound_id,
                int_dtype=tf.int32):
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

    # Broadcast rules for bound_t:
    # ============================
    # If bound_t is a scalar, it is broadcast to [M, 1]
    # If bound_t is a 1-D tensor, it is broadcast to [M, bound_t.shape[0]]
    # If bound_t is a 2-D tensor, leave alone

    bound_t = tf.convert_to_tensor(bound_t)
    if bound_t.shape.rank == 0:
        bound_t = tf.broadcast_to(bound_t, [events.shape[0], 1])
    elif bound_t.shape.rank == 1:
        bound_t = tf.broadcast_to(bound_t, [events.shape[0], bound_t.shape[0]])

    def true_fn():
        # Fetch events
        target_events = tf.gather(events, target_id, axis=-1)  # MxT
        bound_events = tf.gather(events, bound_id, axis=-1)  # MxT

        # Calculate event counting process
        diff = target_events - bound_events
        cumdiff = tf.abs(tf.cumsum(diff, axis=-1))

        # TODO: check the validity of bound_id+1
        bound_init_state = tf.gather(
            initial_state,
            target_id + tf.cast(bound_id > target_id, dtype=int_dtype),
            axis=-1)

        indices = tf.stack([
            tf.repeat(tf.range(cumdiff.shape[0], dtype=int_dtype),
                      [bound_t.shape[1]]),
            tf.reshape(bound_t, [-1])
            ], axis=-1)
        indices = tf.reshape(indices, [cumdiff.shape[0], bound_t.shape[1], 2])
        free_events = tf.gather_nd(cumdiff, indices) + bound_init_state[:, None]
        return free_events

    # Manual broadcasting of n_events_t is required here so that the XLA
    # compiler can guarantee that the output shapes of true_fn() and
    # false_fn() are equal.
    def false_fn():
        return int_dtype.max * tf.ones([events.shape[0]] + [bound_t.shape[1]],
                                       dtype=events.dtype)

    ret_val = tf.cond(bound_id != -1, true_fn, false_fn)
    return ret_val


class Deterministic2(tfd.Deterministic):
    def __init__(self,
                 loc,
                 atol=None,
                 rtol=None,
                 validate_args=False,
                 allow_nan_stats=True,
                 log_prob_dtype=tf.float32,
                 name='Deterministic'):
        parameters = dict(locals())
        super(Deterministic2, self).__init__(
            loc,
            atol=atol,
            rtol=rtol,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name
        )
        self.log_prob_dtype = log_prob_dtype

    def _prob(self, x):
        return tf.constant(1, dtype=self.log_prob_dtype)


def TimeDelta(dmax, name=None):
    outcomes = tf.concat([-tf.range(1, dmax + 1), tf.range(1, dmax + 1)],
                         axis=0)
    logits = tf.ones_like(outcomes, dtype=tf.float64)
    return tfd.FiniteDiscrete(outcomes=outcomes, logits=logits, name=name)


def EventTimeProposal(events, initial_state, topology, d_max, n_max,
                      dtype=tf.int32, name=None):
    """Draws an event time move proposal.
    :param events: a [M, T, K] tensor of event times (M number of metapopulations,
                  T number of timepoints, K number of transitions)
    :param initial_state: a [M, S] tensor of initial metapopulation x state counts
    :param topology: a 3-element tuple of (previous_transition, target_transition,
                                           next_transition), eg "(s->e, e->i, i->r)"
                                           (assuming we are interested presently in e->i, `None` for boundaries)
    :param d_max: the maximum distance over which to move (in time)
    :param n_max: the maximum number of events to move
    """
    target_events = tf.gather(events, topology.target, axis=-1)
    time_interval = tf.range(d_max, dtype=dtype)

    def t_():
        x = tf.cast(target_events > 0, dtype=tf.float64)
        logits = tf.math.log(x)
        return tfd.Multinomial(total_count=1, logits=tf.math.log(x), name='t_')

    def t(t_):
        #x = tf.cast(target_events > 0, dtype=tf.float64)  # [M, T]
        #return tfd.Categorical(logits=tf.math.log(x), name='event_coords')
        return Deterministic2(
            tf.argmax(t_, axis=-1, output_type=dtype),
            log_prob_dtype=events.dtype,
            name='t')

    def delta_t():
        return TimeDelta(d_max, name='TimeDelta')

    def x_star(t, delta_t):
        # Compute bounds
        # The limitations of XLA mean that we must calculate bounds for
        # intervals [t, t+delta_t) if delta_t > 0, and [t+delta_t, t) if
        # delta_t is < 0.
        t = t[..., tf.newaxis]
        bound_interval = tf.where(delta_t < 0,
                                  t - time_interval - 1,  # [t+delta_t, t)
                                  t + time_interval)  # [t, t+delta_t)

        bound_event_id = tf.where(delta_t < 0,
                                  topology.prev or -1,
                                  topology.next or -1)

        free_events = _abscumdiff(events=events, initial_state=initial_state,
                                  target_id=topology.target,
                                  bound_t=bound_interval,
                                  bound_id=bound_event_id, int_dtype=dtype)
        # Mask out bits of the interval we don't need for our delta_t
        inf_mask = tf.cumsum(tf.one_hot(tf.math.abs(delta_t),
                                        d_max,
                                        on_value=dtype.max,
                                        dtype=events.dtype))
        free_events = tf.maximum(inf_mask, free_events)
        free_events = tf.reduce_min(free_events, axis=-1)
        indices = tf.stack([
            tf.range(events.shape[0], dtype=dtype),
            t[:, 0]], axis=-1)
        available_events = tf.gather_nd(target_events, indices)
        max_events = tf.minimum(free_events, available_events)
        max_events = tf.clip_by_value(max_events, clip_value_min=0,
                                      clip_value_max=n_max)

        # Draw x_star
        # Todo Lower bound must be 1.  We must *always* move something.
        return UniformInteger(low=0, high=max_events + 1, name='x_star')

    return tfd.JointDistributionNamed(dict(t_=t_,
                                           t=t,
                                           delta_t=delta_t,
                                           x_star=x_star), name=name)


def FilteredEventTimeProposal(events, initial_state, topology, d_max, n_max,
                              dtype=tf.int32, name=None):
    """FilteredEventTimeProposal allows us to choose a subset of indices
    in `range(events.shape[0])` for which to propose an update.  The
    results are then broadcast back to `events.shape[0]`. """
    target_events = tf.gather(events, topology.target, axis=-1)

    def m():
        hot_meta = tf.math.count_nonzero(target_events, axis=1,
                                         keepdims=True) > 0
        hot_meta = tf.reshape(hot_meta, [1, events.shape[0]])
        logits = tf.math.log(tf.cast(hot_meta, tf.float64))
        X = tfd.Categorical(logits=logits, name='m')
        return X

    def move(m):
        """We select out meta-population `m` from the first
        dimension of `events`.
        :param m: a 1-D tensor of indices of meta-populations
        :return: a random variable of type `EventTimeProposal`
        """
        select_meta = tf.gather(events, m, axis=0)
        select_init = tf.gather(initial_state, m, axis=0)
        return EventTimeProposal(select_meta, select_init, topology, d_max,
                                 n_max, dtype=dtype, name=None)

    return tfd.JointDistributionNamed(dict(m=m,
                                           move=move))
