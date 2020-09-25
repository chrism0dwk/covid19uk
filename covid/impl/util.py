"""Utility functions for model implementation code"""

import numpy as np
import tensorflow as tf
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
from tensorflow_probability.python.internal import prefer_static as ps


def which(predicate):
    """Returns the indices of True elements of predicate"""
    with tf.name_scope("which"):
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

    i_shp = state_shape[:-1] + [trm_coords.shape[0]] + [len(state_shape) + 1]

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
    rate_tensor = tf.scatter_nd(
        indices=indices, updates=rates, shape=output_shape, name="build_markov_matrix"
    )
    return rate_tensor


def compute_state(initial_state, events, stoichiometry):
    """Computes a state tensor from initial state and event tensor

    :param initial_state: a tensor of shape [M, S]
    :param events: a tensor of shape [M, T, X]
    :param stoichiometry: a stoichiometry matrix of shape [X, S] describing
                          how transitions update the state.
    :return: a tensor of shape [M, T, S] describing the state of the
             system for each batch M at time T.
    """
    if isinstance(stoichiometry, tf.Tensor):
        stoichiometry_ = tf.cast(stoichiometry, dtype=events.dtype)
    else:
        stoichiometry_ = tf.convert_to_tensor(stoichiometry, dtype=events.dtype)
    increments = tf.tensordot(events, stoichiometry_, axes=[[-1], [-2]])  # mtx,xs->mts
    cum_increments = tf.cumsum(increments, axis=-2, exclusive=True)
    state = cum_increments + tf.expand_dims(initial_state, axis=-2)
    return state


def transition_coords(stoichiometry):
    src = np.where(stoichiometry == -1)[1]
    dest = np.where(stoichiometry == 1)[1]
    return np.stack([src, dest], axis=-1)


def batch_gather(tensor, indices):
    """Written by Chris Suter (c) 2020
       Modified by Chris Jewell, 2020
    """
    tensor_shape = ps.shape(tensor)  # B + E
    tensor_rank = ps.rank(tensor)
    indices_shape = ps.shape(indices)  # [N, E]
    num_outputs = indices_shape[0]
    non_batch_rank = indices_shape[1]  # r(E)
    batch_rank = tensor_rank - non_batch_rank

    # batch_shape = tf.cast(tensor_shape[:batch_rank], dtype=tf.int64)
    # batch_size = tf.reduce_prod(batch_shape)
    # Create indices into batch_shape, of shape [batch_size, batch_rank]
    # batch_indices = tf.transpose(
    #    tf.unravel_index(tf.range(batch_size), dims=batch_shape)
    # )

    batch_shape = tensor_shape[:batch_rank]
    batch_size = np.prod(batch_shape)
    batch_indices = np.transpose(
        np.unravel_index(np.arange(batch_size), dims=batch_shape)
    )

    # Tile the batch indices num_outputs times
    batch_indices_tiled = tf.reshape(
        tf.tile(batch_indices, multiples=[1, num_outputs]),
        [batch_size * num_outputs, -1],
    )

    batched_output_indices = tf.tile(indices, multiples=[batch_size, 1])
    full_indices = tf.concat([batch_indices_tiled, batched_output_indices], axis=-1)

    output_shape = ps.concat([batch_shape, [num_outputs]], axis=0)
    return tf.reshape(tf.gather_nd(tensor, full_indices), output_shape)
