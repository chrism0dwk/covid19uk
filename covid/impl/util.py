"""Utility functions for model implementation code"""

import numpy as np
import tensorflow as tf

def _gen_index(state, trm_coords):
    """Returns a tensor of indices indexing
    coordinates trm_coords into a
    state.shape + state.shape[-1] tensor.
    """
    state = tf.convert_to_tensor(state)
    trm_coords = tf.convert_to_tensor(trm_coords)

    output_shape = state.shape.as_list() + [state.shape[-1]]
    i_shp = state.shape[:-1].as_list() + [trm_coords.shape[0]] + \
            [len(state.shape) + 1]
    num_updates = tf.reduce_prod(i_shp[:-1], axis=0)

    b_idx = np.array(list(np.ndindex(*i_shp[:-1])))[:, :-1]
    m_idx = tf.tile(trm_coords, [tf.reduce_prod(i_shp[:-2]), 1])

    idx = tf.concat([b_idx, m_idx], axis=-1)
    return tf.reshape(idx, i_shp)


def make_transition_rate_matrix(rates, rate_coords, state):
    """Create a transition rate matrix
    :param rates: batched transition rate tensors
    :param rate_coords: coordinates of rates in resulting transition matrix
    :param state: the state tensor with ns states
    :returns: a tensor of shape [..., ns, ns]
    """
    indices = _gen_index(state, rate_coords)
    updates = tf.stack(rates, axis=-1)
    output_shape = state.shape.as_list() + [state.shape[-1]]
    rate_tensor = tf.scatter_nd(indices=indices,
                                updates=updates,
                                shape=output_shape)
    return rate_tensor