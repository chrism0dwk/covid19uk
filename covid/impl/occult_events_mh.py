from collections import namedtuple
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.util import SeedStream

from covid import config
from covid.impl.event_time_proposal import TransitionTopology, FilteredEventTimeProposal
from covid.impl.occult_proposal import AddOccultProposal, DelOccultProposal

tfd = tfp.distributions
DTYPE = config.floatX


OccultKernelResults = namedtuple(
    "KernelResults", ("log_acceptance_correction", "target_log_prob", "extra")
)


def _nonzero_rows(m):
    return tf.cast(tf.reduce_sum(m, axis=-1) > 0.0, m.dtype)


def _maybe_expand_dims(x):
    """If x is a scalar, give it at least 1 dimension"""
    x = tf.convert_to_tensor(x)
    if x.shape == ():
        return tf.expand_dims(x, axis=0)
    return x


def _add_events(events, m, t, x, x_star):
    """Adds `x_star` events to metapopulation `m`,
       time `t`, transition `x` in `events`."""
    x = _maybe_expand_dims(x)
    indices = tf.stack([m, t, x], axis=-1)
    return tf.tensor_scatter_nd_add(events, indices, x_star)


class UncalibratedOccultUpdate(tfp.mcmc.TransitionKernel):
    """UncalibratedEventTimesUpdate"""

    def __init__(
        self,
        target_log_prob_fn,
        target_event_id,
        nmax,
        t_range=None,
        seed=None,
        name=None,
    ):
        """An uncalibrated random walk for event times.
        :param target_log_prob_fn: the log density of the target distribution
        :param target_event_id: the position in the last dimension of the events
                                tensor that we wish to move
        :param t_range: a tuple containing earliest and latest times between which 
                        to update occults.
        :param seed: a random seed
        :param name: the name of the update step
        """
        self._target_log_prob_fn = target_log_prob_fn
        self._seed_stream = SeedStream(seed, salt="UncalibratedOccultUpdate")
        self._name = name
        self._parameters = dict(
            target_log_prob_fn=target_log_prob_fn,
            target_event_id=target_event_id,
            nmax=nmax,
            t_range=t_range,
            seed=seed,
            name=name,
        )
        self.tx_topology = TransitionTopology(None, target_event_id, None)

    @property
    def target_log_prob_fn(self):
        return self._parameters["target_log_prob_fn"]

    @property
    def target_event_id(self):
        return self._parameters["target_event_id"]

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
        with tf.name_scope("occult_rw/onestep"):

            def true_fn():
                with tf.name_scope("true_fn"):
                    proposal = AddOccultProposal(
                        events=current_events,
                        n_max=self.parameters["nmax"],
                        t_range=self.parameters["t_range"],
                    )
                    update = proposal.sample()
                    next_state = _add_events(
                        events=current_events,
                        m=update["m"],
                        t=update["t"],
                        x=self.tx_topology.target,
                        x_star=tf.cast(update["x_star"], current_events.dtype),
                    )
                    reverse = DelOccultProposal(next_state, self.tx_topology)
                    q_fwd = tf.reduce_sum(proposal.log_prob(update))
                    q_rev = tf.reduce_sum(reverse.log_prob(update))
                    log_acceptance_correction = q_rev - q_fwd
                return update, next_state, log_acceptance_correction

            def false_fn():
                with tf.name_scope("false_fn"):
                    proposal = DelOccultProposal(current_events, self.tx_topology)
                    update = proposal.sample()
                    next_state = _add_events(
                        events=current_events,
                        m=update["m"],
                        t=update["t"],
                        x=[self.tx_topology.target],
                        x_star=tf.cast(-update["x_star"], current_events.dtype),
                    )
                    reverse = AddOccultProposal(
                        events=next_state,
                        n_max=self.parameters["nmax"],
                        t_range=self.parameters["t_range"],
                    )
                    q_fwd = tf.reduce_sum(proposal.log_prob(update))
                    q_rev = tf.reduce_sum(reverse.log_prob(update))
                    log_acceptance_correction = q_rev - q_fwd

                return update, next_state, log_acceptance_correction

            u = tfd.Uniform().sample()
            delta, next_state, log_acceptance_correction = tf.cond(
                u < 0.5, true_fn, false_fn
            )
            # tf.debugging.assert_non_negative(
            #     next_state, message="Negative occults occurred"
            # )

            next_target_log_prob = self.target_log_prob_fn(next_state)
            return [
                next_state,
                OccultKernelResults(
                    log_acceptance_correction=log_acceptance_correction,
                    target_log_prob=next_target_log_prob,
                    extra=tf.concat([delta["m"], delta["t"], delta["x_star"]], axis=0),
                ),
            ]

    def bootstrap_results(self, init_state):
        with tf.name_scope("uncalibrated_event_times_rw/bootstrap_results"):
            init_state = tf.convert_to_tensor(init_state, dtype=DTYPE)
            init_target_log_prob = self.target_log_prob_fn(init_state)
            return OccultKernelResults(
                log_acceptance_correction=tf.constant(0.0, dtype=DTYPE),
                target_log_prob=init_target_log_prob,
                extra=tf.constant([0, 0, 0], dtype=tf.int32),
            )
