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
    def propagate_fn(t, state):
        rate_matrix = h(t, state)
        # Set diagonal to be the negative of the sum of other elements in each row
        rate_matrix = tf.linalg.set_diag(rate_matrix,
                                         -tf.reduce_sum(rate_matrix, axis=-1))
        # Calculate Markov transition probability matrix
        markov_transition = tf.linalg.expm(rate_matrix*time_step)
        # Sample new state
        new_state = tfd.Multinomial(total_count=state,
                                    probs=markov_transition).sample()
        new_state = tf.reduce_sum(new_state, axis=-1)
        return new_state
    return propagate_fn


@tf.function(autograph=False)  # Algorithm runs super slow if uncommented.  Weird!
def chain_binomial_simulate(hazard_fn, state, start, end, time_step):
    propagate = chain_binomial_propagate(hazard_fn, time_step)
    times = tf.range(start, end, time_step)

    output = tf.TensorArray(state.dtype, size=times.shape[0])
    output = output.write(0, state)

    cond = lambda i, *_: i < times.shape[0]
    def body(i, state, output):
      state = propagate(i, state)
      output = output.write(i, state)
      return i + 1, state, output
    _, state, output = tf.while_loop(cond, body, loop_vars=(0, state, output))
    return times, output.stack()
