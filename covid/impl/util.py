"""Utility functions for model implementation code"""

import numpy as np
import tensorflow as tf
from tensorflow_probability.python.mcmc.internal import util as mcmc_util


def which(predicate):
    """Returns the indices of True elements of predicate"""
    with tf.name_scope('which'):
        x = tf.cast(predicate, dtype=tf.int32)
        index_range = tf.range(x.shape[0])
        indices = tf.cumsum(x) * x
        indices = tf.scatter_nd(indices[:, None], index_range, x.shape)
        return indices[1:]


def _gen_index(state_shape, trm_coords):
    """Returns a tensor of indices indexing
    coordinates trm_coords into a
    state_shape + state_shape[-1] tensor.
    """
    trm_coords = tf.convert_to_tensor(trm_coords)

    i_shp = state_shape[:-1] + [trm_coords.shape[0]] + \
            [len(state_shape) + 1]

    b_idx = np.array(list(np.ndindex(*i_shp[:-1])))[:, :-1]
    m_idx = tf.tile(trm_coords, [tf.reduce_prod(i_shp[:-2]), 1])

    idx = tf.concat([b_idx, m_idx], axis=-1)
    return tf.reshape(idx, i_shp)


def make_transition_matrix(rates, rate_coords, state_shape):
    """Create a transition rate matrix
    :param rates: batched transition rate tensors  [b1, b2, n_rates] or a list of length n_rates of batched
                  tensors [b1, b2]
    :param rate_coords: coordinates of rates in resulting transition matrix
    :param state_shape: the shape of the state tensor with ns states
    :returns: a tensor of shape [..., ns, ns]
    """
    indices = _gen_index(state_shape, rate_coords)
    if mcmc_util.is_list_like(rates):
        rates = tf.stack(rates, axis=-1)
    output_shape = state_shape + [state_shape[-1]]
    rate_tensor = tf.scatter_nd(indices=indices,
                                updates=rates,
                                shape=output_shape)
    return rate_tensor


