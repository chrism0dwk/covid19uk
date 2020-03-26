"""Functions for chain binomial simulation."""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


def chain_binomial_propagate(h, time_step):
    """Propagates the state of a population according to discrete time dynamics.

    :param h: a hazard rate function returning the non-row-normalised Markov transition rate matrix
    :param time_step: the time step
    :returns : a function that propagate `state[t]` -> `state[t+time_step]`
    """
    def propagate_fn(state):
        # State is assumed to be of shape [s, n] where s is the number of states
        #   and n is the number of population strata.
        # TODO: having state as [s, n] means we have to do some funky transposition.  It may be better
        #       to have state.shape = [n, s] which avoids transposition below, but may lead to slower
        #       rate calculations.
        rate_matrix = h(state)
        rate_matrix = tf.transpose(rate_matrix, perm=[2, 0, 1])
        # Set diagonal to be the negative of the sum of other elements in each row
        rate_matrix = tf.linalg.set_diag(rate_matrix, -tf.reduce_sum(rate_matrix, axis=2))
        # Calculate Markov transition probability matrix
        markov_transition = tf.linalg.expm(rate_matrix*time_step)
        # Sample new state
        new_state = tfd.Multinomial(total_count=tf.transpose(state),
                                    probs=markov_transition).sample()
        new_state = tf.reduce_sum(new_state, axis=1)
        return tf.transpose(new_state)
    return propagate_fn


def chain_binomial_simulate(hazard_fn, state, start, end, time_step):

    propagate = chain_binomial_propagate(hazard_fn, time_step)
    times = tf.range(start, end, time_step)

    output = tf.TensorArray(tf.float64, size=times.shape[0])
    output = output.write(0, state)

    for i in tf.range(1, times.shape[0]):
        state = propagate(state)
        output = output.write(i, state)

    sim = output.gather(tf.range(times.shape[0]))
    return times, sim


