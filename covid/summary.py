"""Epidemic summary measure functions"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from gemlib_tfp_extra.util import compute_state


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


def power_iteration(A, tol=1e-3):
    b_k = tf.random.normal([A.shape[1], 1], dtype=A.dtype)
    epsilon = tf.constant(1.0, dtype=A.dtype)
    i = 0
    while tf.greater(epsilon, tol):
        b_k1 = tf.matmul(A, b_k)
        b_k1_norm = tf.linalg.norm(b_k1)
        b_k_new = b_k1 / b_k1_norm
        epsilon = tf.reduce_sum(tf.pow(b_k_new - b_k, 2))
        b_k = b_k_new
        i += 1
    return b_k, i


def rayleigh_quotient(A, b):
    b = tf.squeeze(b)
    numerator = tf.einsum("...i,...i->...", b, tf.linalg.matvec(A, b))
    denominator = tf.einsum("...i,...i->...", b, b)
    return numerator / denominator

