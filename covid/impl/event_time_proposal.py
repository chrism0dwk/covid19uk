"""Mechanism for proposing event times to move"""
import collections
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

TransitionTopology = collections.namedtuple('TransitionTopology',
                                            ('prev',
                                             'target',
                                             'next'))

def _abscumdiff(events, initial_state,
                target_t, target_id,
                bound_t, bound_id):
    """Returns the number of free events to move in target_events
       bounded by max([N_{target_id}(t)-N_{bound_id}(t)]_{bound_t}).
    :param events: a [(M), T, X] tensor of transition events
    :param initial_state: a [M, X] tensor of the constraining initial state
    :param target_t: the target time
    :param target_id: the Xth index of the target event
    :param bound_t: the times to compute the constraints
    :param bound_id: the Xth index of the bounding event, -1 implies no bound
    :returns: a tensor of shape [M] + bound_t.shape[0] +  of max free events,
              dtype=target_events.dtype
    """

    bound_t = tf.convert_to_tensor(bound_t)
    if bound_t.ndim == 0:
        bound_t = tf.expand_dims(bound_t, axis=0)
    target_t = tf.convert_to_tensor(target_t)
    if target_t.ndim == 0:
        target_t = tf.expand_dims(target_t, axis=0)

    def true_fn():
        # Fetch events
        target_events = tf.gather(events, target_id, axis=-1)  # MxT
        bound_events = tf.gather(events, bound_id, axis=-1)  # MxT

        # Calculate event counting process
        diff = target_events - bound_events
        cumdiff = tf.abs(tf.cumsum(diff, axis=-1))

        # TODO: check the validity of bound_id+1
        bound_init_state = tf.gather(initial_state,
                                     target_id + tf.cast(bound_id > target_id, dtype=tf.int32),
                                     axis=-1)

        indices = tf.stack([
            tf.repeat(tf.range(bound_t.shape[0], dtype=tf.int64), [bound_t.shape[1]]),
            tf.reshape(bound_t, [-1])
        ], axis=-1)
        indices = tf.reshape(indices, bound_t.shape + [2])
        free_events = tf.gather_nd(cumdiff, indices) + bound_init_state[:, None]
        return free_events

    # Manual broadcasting of n_events_t is required here so that the XLA
    # compiler can guarantee that the output shapes of true_fn() and
    # false_fn() are equal.
    def false_fn():
        return tf.zeros([events.shape[0]] + [bound_t.shape[0]])

    ret_val = tf.cond(bound_id != -1, true_fn, false_fn)
    return ret_val


def TimeDelta(dmax, p_pos=0.5, name=None):
    def u():
        return tfd.Bernoulli(probs=p_pos)

    def magnitude():
        return tfd.Categorical(logits=tf.ones(dmax))  # Issue if dmax unknown at compile time

    def delta_t(u, magnitude):
        return tfd.Deterministic((2 * u - 1) * (magnitude+1), name='delta_t')

    return tfd.JointDistributionNamed(dict(u=u,
                                           magnitude=magnitude,
                                           delta_t=delta_t), name=name)


def EventTimeProposal(events, initial_state, topology, d_max, n_max, name=None):
    """Draws an event time move proposal.
    :param events: a [M, T, K] tensor of event times (M number of metapopulations,
                  T number of timepoints, K number of transitions)
    :param topology: a 3-element tuple of (previous_transition, target_transition,
                                           next_transition)
    :param d_max: the maximum distance over which to move
    :param n_max: the maximum number of events to move
    """
    target_events = tf.gather(events, topology.target, axis=-1)
    time_interval = tf.range(d_max, dtype=tf.int64)

    def one_hot_rows():
        # OneHotCategorical on each batch
        x = tf.cast(target_events > 0, dtype=tf.float32)
        logits = tf.math.log(x)
        return tfd.OneHotCategorical(logits=logits, name='one_hot_rows')

    def t(one_hot_rows):
        col_idx = tf.argmax(one_hot_rows, axis=-1)
        return tfd.Deterministic(col_idx[..., None], name='event_coords')

    def time_delta():
        return TimeDelta(d_max, name='TimeDelta')

    def x_star(t, time_delta):
        delta_t = time_delta['delta_t']
        # Compute bounds
        # The limitations of XLA mean that we must calculate bounds for
        # intervals [t, t+delta_t) if delta_t > 0, and [t+delta_t, t) if
        # delta_t is < 0.
        bound_interval = tf.where(delta_t < 0,
                                  t - time_interval - 1,  # [t+delta_t, t)
                                  t + time_interval)  # [t, t+delta_t)

        bound_event_id = tf.where(delta_t < 0,
                                  topology.prev or -1,
                                  topology.next or -1)

        free_events = _abscumdiff(events=events, initial_state=initial_state,
                                  target_t=t,
                                  target_id=topology.target,
                                  bound_t=bound_interval,
                                  bound_id=bound_event_id)
        # Mask out bits of the interval we don't need for our delta_t
        inf_mask = tf.cumsum(tf.one_hot(tf.math.abs(delta_t),
                                        d_max, dtype=tf.int32)) * tf.int32.max
        free_events = tf.reduce_min(tf.cast(inf_mask,
                                            free_events.dtype) + free_events, axis=-1)
        indices = tf.stack([
            tf.range(events.shape[0], dtype=tf.int64),
            tf.squeeze(t)
        ], axis=-1)
        available_events = tf.gather_nd(events[..., topology.target], indices)
        max_events = tf.minimum(free_events, available_events)
        max_events = tf.clip_by_value(max_events, clip_value_min=0,
                                      clip_value_max=n_max)

        # Draw x_star
        return tfd.Uniform(low=0, high=max_events, name='x_star')

    return tfd.JointDistributionNamed(dict(one_hot_rows=one_hot_rows,
                                           t=t,
                                           time_delta=time_delta,
                                           x_star=x_star), name=name)
