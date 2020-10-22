from collections import namedtuple
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.util import SeedStream

from covid import config
from covid.impl.event_time_proposal import TransitionTopology, FilteredEventTimeProposal

tfd = tfp.distributions
DTYPE = config.floatX


EventTimesKernelResults = namedtuple(
    "KernelResults", ("log_acceptance_correction", "target_log_prob", "extra")
)


def _is_within(x, low, high):
    """Returns true if low <= x < high"""
    return tf.logical_and(tf.less_equal(low, x), tf.less(x, high))


def _nonzero_rows(m):
    return tf.cast(tf.reduce_sum(m, axis=-1) > 0.0, m.dtype)


def _move_events(event_tensor, event_id, m, from_t, to_t, n_move):
    """Subtracts n_move from event_tensor[m, from_t, event_id]
    and adds n_move to event_tensor[m, to_t, event_id].

    :param event_tensor: shape [M, T, X]
    :param event_id: the event id to move
    :param m: the metapopulation to move
    :param from_t: the move-from time
    :param to_t: the move-to time
    :param n_move: the number of events to move
    :return: the modified event_tensor
    """
    # Todo rationalise this -- compute a delta, and add once.
    indices = tf.stack(
        [m, from_t, tf.broadcast_to(event_id, m.shape)], axis=-1  # All meta-populations
    )  # Event
    # Subtract x_star from the [from_t, :, event_id] row of the state tensor
    n_move = tf.cast(n_move, event_tensor.dtype)
    new_state = tf.tensor_scatter_nd_sub(event_tensor, indices, n_move)
    indices = tf.stack([m, to_t, tf.broadcast_to(event_id, m.shape)], axis=-1)
    # Add x_star to the [to_t, :, event_id] row of the state tensor
    new_state = tf.tensor_scatter_nd_add(new_state, indices, n_move)
    return new_state


def _reverse_move(move):
    move["t"] = move["t"] + move["delta_t"]
    move["delta_t"] = -move["delta_t"]
    return move


class UncalibratedEventTimesUpdate(tfp.mcmc.TransitionKernel):
    """UncalibratedEventTimesUpdate"""

    def __init__(
        self,
        target_log_prob_fn,
        target_event_id,
        prev_event_id,
        next_event_id,
        initial_state,
        dmax,
        mmax,
        nmax,
        name=None,
    ):
        """An uncalibrated random walk for event times.
        :param target_log_prob_fn: the log density of the target distribution
        :param target_event_id: the position in the first dimension of the events
                                tensor that we wish to move
        :param prev_event_id: the position of the previous event in the events tensor
        :param next_event_id: the position of the next event in the events tensor
        :param initial_state: the initial state tensor
        :param seed: a random seed
        :param name: the name of the update step
        """
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
            name=name,
        )
        self.tx_topology = TransitionTopology(
            prev_event_id, target_event_id, next_event_id
        )
        self.time_offsets = tf.range(self.parameters["dmax"])

    @property
    def target_log_prob_fn(self):
        return self._parameters["target_log_prob_fn"]

    @property
    def target_event_id(self):
        return self._parameters["target_event_id"]

    @property
    def prev_event_id(self):
        return self._parameters["prev_event_id"]

    @property
    def next_event_id(self):
        return self._parameters["next_event_id"]

    @property
    def seed(self):
        return self._parameters["seed"]

    @property
    def name(self):
        return self._parameters["name"]

    @property
    def parameters(self):
        """Return `dict` of ``__init__`` arguments and their values."""
        return self._parameters

    @property
    def is_calibrated(self):
        return False

    def one_step(self, current_events, previous_kernel_results, seed=None):
        """One update of event times.
        :param current_events: a [T, M, X] tensor containing number of events
                               per time t, metapopulation m,
                               and transition x.
        :param previous_kernel_results: an object of type
                                        UncalibratedRandomWalkResults.
        :returns: a tuple containing new_state and UncalibratedRandomWalkResults
        """
        with tf.name_scope("uncalibrated_event_times_rw/onestep"):
            target_events = current_events[..., self.tx_topology.target]
            num_times = target_events.shape[1]

            proposal = FilteredEventTimeProposal(
                events=current_events,
                initial_state=self.parameters["initial_state"],
                topology=self.tx_topology,
                m_max=self.parameters["mmax"],
                d_max=self.parameters["dmax"],
                n_max=self.parameters["nmax"],
            )
            update = proposal.sample(seed=seed)

            move = update["move"]
            to_t = move["t"] + move["delta_t"]

            def true_fn():
                with tf.name_scope("true_fn"):
                    # Prob of fwd move
                    q_fwd = proposal.log_prob(update)
                    tf.debugging.assert_all_finite(q_fwd, "q_fwd is not finite")

                    # Propagate state
                    next_state = _move_events(
                        event_tensor=current_events,
                        event_id=self.tx_topology.target,
                        m=update["m"],
                        from_t=move["t"],
                        to_t=to_t,
                        n_move=move["x_star"],
                    )

                    next_target_log_prob = self.target_log_prob_fn(next_state)

                    # Calculate proposal mass ratio
                    rev_move = _reverse_move(move.copy())
                    rev_update = dict(m=update["m"], move=rev_move)
                    Q_rev = FilteredEventTimeProposal(  # pylint: disable-invalid-name
                        events=next_state,
                        initial_state=self.parameters["initial_state"],
                        topology=self.tx_topology,
                        m_max=self.parameters["mmax"],
                        d_max=self.parameters["dmax"],
                        n_max=self.parameters["nmax"],
                    )

                    # Prob of reverse move and q-ratio
                    q_rev = Q_rev.log_prob(rev_update)
                    log_acceptance_correction = tf.reduce_sum(q_rev - q_fwd)

                    return (
                        next_target_log_prob,
                        log_acceptance_correction,
                        next_state,
                    )

            def false_fn():
                with tf.name_scope("false_fn"):
                    next_target_log_prob = tf.constant(
                        -np.inf, dtype=current_events.dtype
                    )
                    log_acceptance_correction = tf.constant(
                        0.0, dtype=current_events.dtype
                    )
                    return (
                        next_target_log_prob,
                        log_acceptance_correction,
                        current_events,
                    )

            # Trap out-of-bounds moves that go outside [0, num_times)
            next_target_log_prob, log_acceptance_correction, next_state = tf.cond(
                tf.reduce_all(_is_within(to_t, 0, num_times)),
                true_fn=true_fn,
                false_fn=false_fn,
            )

            x_star_results = tf.scatter_nd(
                update["m"][:, tf.newaxis],
                tf.abs(move["x_star"] * move["delta_t"]),
                [current_events.shape[0]],
            )

            return [
                next_state,
                EventTimesKernelResults(
                    log_acceptance_correction=log_acceptance_correction,
                    target_log_prob=next_target_log_prob,
                    extra=tf.cast(x_star_results, current_events.dtype),
                ),
            ]

    def bootstrap_results(self, init_state):
        with tf.name_scope("uncalibrated_event_times_rw/bootstrap_results"):
            init_state = tf.convert_to_tensor(init_state, dtype=DTYPE)
            init_target_log_prob = self.target_log_prob_fn(init_state)
            return EventTimesKernelResults(
                log_acceptance_correction=tf.constant(0.0, dtype=DTYPE),
                target_log_prob=init_target_log_prob,
                extra=tf.zeros(init_state.shape[-3], dtype=DTYPE),
            )
