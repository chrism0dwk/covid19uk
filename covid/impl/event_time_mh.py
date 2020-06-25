from pprint import pprint

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.util import SeedStream

from covid import config
from covid.impl.event_time_proposal import TransitionTopology, FilteredEventTimeProposal
from covid.impl.mcmc import KernelResults

tfd = tfp.distributions
DTYPE = config.floatX


def _is_within(x, low, high):
    """Returns true if low <= x < high"""
    return tf.logical_and(tf.less_equal(low, x), tf.less(x, high))


def _nonzero_rows(m):
    return tf.cast(tf.reduce_sum(m, axis=-1) > 0.0, m.dtype)


def _max_free_events(
    events, initial_state, target_t, target_id, constraint_t, constraint_id
):
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
        constraining_events = tf.gather(events, constraint_id, axis=-1)  # TxM
        constraining_cumsum = tf.cumsum(constraining_events, axis=0)  # TxM
        constraining_init_state = tf.gather(initial_state, constraint_id + 1, axis=-1)
        n1 = tf.gather(target_cumsum, constraint_t, axis=0)
        n2 = tf.gather(constraining_cumsum, constraint_t, axis=0)
        free_events = tf.abs(n1 - n2) + constraining_init_state
        max_free_events = tf.minimum(
            free_events, tf.gather(target_events_, target_t, axis=0)
        )
        return max_free_events

    # Manual broadcasting of n_events_t is required here so that the XLA
    # compiler can guarantee that the output shapes of true_fn() and
    # false_fn() are equal.  Known shape information can thus be
    # propagated right through the algorithm, so the return value has known shape.
    def false_fn():
        n_events_t = tf.gather(events[..., target_id], target_t, axis=0)
        return tf.broadcast_to(
            [n_events_t], [constraint_t.shape[0]] + [n_events_t.shape[0]]
        )

    ret_val = tf.cond(constraint_id != -1, true_fn, false_fn)
    return ret_val


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


class EventTimesUpdate(tfp.mcmc.TransitionKernel):
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
        seed=None,
        name=None,
    ):
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
        self._seed_stream = SeedStream(seed, salt="EventTimesUpdate")
        self._impl = tfp.mcmc.MetropolisHastings(
            inner_kernel=UncalibratedEventTimesUpdate(
                target_log_prob_fn=target_log_prob_fn,
                target_event_id=target_event_id,
                prev_event_id=prev_event_id,
                next_event_id=next_event_id,
                dmax=dmax,
                mmax=mmax,
                nmax=nmax,
                initial_state=initial_state,
            )
        )
        self._parameters = self._impl.inner_kernel.parameters.copy()
        self._parameters["seed"] = seed

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
        with tf.name_scope("EventTimesUpdate/one_step"):
            next_state, kernel_results = self._impl.one_step(
                current_state, previous_kernel_results
            )
            return next_state, kernel_results

    def bootstrap_results(self, init_state):
        with tf.name_scope("EventTimesUpdate/bootstrap_results"):
            kernel_results = self._impl.bootstrap_results(init_state)
            return kernel_results


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
        seed=None,
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
        self._target_log_prob_fn = target_log_prob_fn
        self._seed_stream = SeedStream(seed, salt="UncalibratedEventTimesUpdate")
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

    def one_step(self, current_events, previous_kernel_results):
        """One update of event times.
        :param current_events: a [T, M, X] tensor containing number of events
                               per time t, metapopulation m,
                               and transition x.
        :param previous_kernel_results: an object of type
                                        UncalibratedRandomWalkResults.
        :returns: a tuple containing new_state and UncalibratedRandomWalkResults
        """
        with tf.name_scope("uncalibrated_event_times_rw/onestep"):
            current_events = tf.transpose(current_events, perm=(1, 0, 2))
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
            update = proposal.sample()

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

                    next_state_tr = tf.transpose(next_state, perm=(1, 0, 2))
                    next_target_log_prob = self._target_log_prob_fn(next_state_tr)

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
                        next_state_tr,
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
                        tf.transpose(current_events, perm=(1, 0, 2)),
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
                KernelResults(
                    log_acceptance_correction=log_acceptance_correction,
                    target_log_prob=next_target_log_prob,
                    extra=tf.cast(x_star_results, current_events.dtype),
                ),
            ]

    def bootstrap_results(self, init_state):
        with tf.name_scope("uncalibrated_event_times_rw/bootstrap_results"):
            init_state = tf.convert_to_tensor(init_state, dtype=DTYPE)
            init_target_log_prob = self.target_log_prob_fn(init_state)
            return KernelResults(
                log_acceptance_correction=tf.constant(0.0, dtype=DTYPE),
                target_log_prob=init_target_log_prob,
                extra=tf.zeros(init_state.shape[-2], dtype=DTYPE),
            )
