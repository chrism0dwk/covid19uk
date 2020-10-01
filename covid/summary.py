"""Epidemic summary measure functions"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from covid.impl.util import compute_state


def mean_and_ci(arr, q=(0.025, 0.975), axis=0, name=None):

    if name is None:
        name = ""
    else:
        name = name + "_"

    q = np.array(q)
    mean = tf.reduce_mean(arr, axis=axis)
    ci = tfp.stats.percentile(arr, q=q * 100.0, axis=axis)

    results = dict()
    results[name + "mean"] = mean
    for i, qq in enumerate(q):
        results[name + str(qq)] = ci[i]
    return results


# Reproduction number calculation
def calc_R_it(theta, xi, events, init_state, covar_data):
    """Calculates effective reproduction number for batches of metapopulations
    :param theta: a tensor of batched theta parameters [B] + theta.shape
    :param xi: a tensor of batched xi parameters [B] + xi.shape
    :param events: a [B, M, T, X] batched events tensor
    :param init_state: the initial state of the epidemic at earliest inference date
    :param covar_data: the covariate data
    :return a batched vector of R_it estimates
    """

    def r_fn(args):
        theta_, xi_, events_ = args
        t = events_.shape[-2] - 1
        state = compute_state(init_state, events_, model_spec.STOICHIOMETRY)
        state = tf.gather(state, t - 1, axis=-2)  # State on final inference day

        par = dict(beta1=theta[0], beta2=theta[1], gamma=theta[2], nu=0.5, xi=xi)

        ngm_fn = model_spec.next_generation_matrix_fn(covar_data, par)
        ngm = ngm_fn(t, state)
        return ngm

    return tf.vectorized_map(r_fn, elems=(theta, xi, events))
