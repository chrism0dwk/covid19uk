"""Utility functions for model implementation code"""

import numpy as np
import tensorflow as tf
from tensorflow_probability.python.mcmc.internal import util as mcmc_util

def _gen_index(state, trm_coords):
    """Returns a tensor of indices indexing
    coordinates trm_coords into a
    state.shape + state.shape[-1] tensor.
    """
    state = tf.convert_to_tensor(state)
    trm_coords = tf.convert_to_tensor(trm_coords)

    i_shp = state.shape[:-1].as_list() + [trm_coords.shape[0]] + \
            [len(state.shape) + 1]

    b_idx = np.array(list(np.ndindex(*i_shp[:-1])))[:, :-1]
    m_idx = tf.tile(trm_coords, [tf.reduce_prod(i_shp[:-2]), 1])

    idx = tf.concat([b_idx, m_idx], axis=-1)
    return tf.reshape(idx, i_shp)


def make_transition_matrix(rates, rate_coords, state):
    """Create a transition rate matrix
    :param rates: batched transition rate tensors  [b1, b2, n_rates] or a list of length n_rates of batched
                  tensors [b1, b2]
    :param rate_coords: coordinates of rates in resulting transition matrix
    :param state: the state tensor with ns states
    :returns: a tensor of shape [..., ns, ns]
    """
    indices = _gen_index(state, rate_coords)
    if mcmc_util.is_list_like(rates):
        rates = tf.stack(rates, axis=-1)
    output_shape = state.shape.as_list() + [state.shape[-1]]
    rate_tensor = tf.scatter_nd(indices=indices,
                                updates=rates,
                                shape=output_shape)
    return rate_tensor


def squared_jumping_distance(chain):
    diff = chain[1:] - chain[:-1]
    diff = diff * diff
    return diff.sum(axis=tuple(np.arange(1, diff.ndim)))
