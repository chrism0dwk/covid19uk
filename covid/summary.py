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
