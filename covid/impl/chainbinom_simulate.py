"""Functions for chain binomial simulation."""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


def chain_binomial_propagate(h, time_step):
    """Propagates the state of a population according to discrete time dynamics.

    :param h: a hazard rate function returning the non-row-normalised Markov transition rate matrix
              This function should return a tensor of dimension [ns, ns, nc] where ns is the number of
              states, and nc is the number of strata within the population.
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
        num_states = markov_transition.shape[-1]
        prev_probs = tf.zeros_like(markov_transition[..., :, 0])
        counts = tf.zeros(markov_transition.shape[:-1].as_list() + [0],
                          dtype=markov_transition.dtype)
        total_count = state
        samples = draw_sample_iterated_binomial_tf_while(
            [],
            num_classes=markov_transition.shape[-1],
            probs=markov_transition,
            num_trials=state,
            seed=42)
        new_state = tf.reduce_sum(samples, axis=-2)

        return new_state
    return propagate_fn


def chain_binomial_simulate(hazard_fn, state, start, end, time_step):
    """Simulates from a discrete time Markov state transition model using multinomial sampling
    across rows of the """
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


def draw_sample_iterated_binomial_tf_while(
    sample_shape, num_classes, probs, num_trials, seed):
  dtype = probs.dtype
  num_trials = tf.cast(num_trials, dtype)
  num_trials += tf.zeros_like(probs[..., 0])  # Pre-broadcast
  def fn(state, accum):
    i, num_trials, consumed_prob = state
    probs_here = tf.gather(probs, i, axis=-1)
    binomial_probs = tf.clip_by_value(probs_here / (1. - consumed_prob), 0, 1)
    binomial = tfd.Binomial(
        total_count=num_trials,
        probs=binomial_probs)
    sample = binomial.sample(sample_shape)
    accum = accum.write(i, sample)
    return (i + 1, num_trials - sample, consumed_prob + probs_here), accum

  i = tf.constant(0)
  consumed_prob = tf.zeros_like(probs[..., 0])
  accum = tf.TensorArray(dtype=dtype, size=probs.shape[-1])
  _, accum = tf.while_loop(cond=lambda state, _: tf.less(state[0], probs.shape[-1]),
                           body=fn,
                           loop_vars=((i, num_trials, consumed_prob), accum))
  stacked = accum.stack()
  indices = tf.range(tf.rank(stacked))
  perm = tf.concat([indices[1:], indices[:1]], axis=0)
  return tf.transpose(accum.stack(), perm=perm)
