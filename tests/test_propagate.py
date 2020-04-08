import unittest

import tensorflow as tf
from covid.impl.discrete_markov import chain_binomial_propagate


class TestPropagate(unittest.TestCase):

    def test_1d_shape(self):
        state = tf.constant([[18], [19], [20]], dtype=tf.float32)
        stoichiometry = tf.constant([[-1, 1, 0], [0, -1, 1]], dtype=tf.float32)
        def h(state):
            return tf.stack([[0.1], [0.1]])
        propagate_fn = chain_binomial_propagate(h, 1, stoichiometry)
        new_state = propagate_fn(state)
        self.assertListEqual(list(new_state.shape), [3, 1])

    def test_2d_shape(self):
        state = tf.constant([[20, 10],
                             [15, 5],
                             [0, 0]], dtype=tf.float32)
        stoichiometry = tf.constant([[-1, 1, 0], [0, -1, 1]], dtype=tf.float32)
        def h(state):
            return tf.stack([[0.1, 0.1],
                             [0.1, 0.1]])
        propagate_fn = chain_binomial_propagate(h, 1, stoichiometry)
        new_state = propagate_fn(state)
        self.assertListEqual(list(new_state.shape), [3, 2])

    def test_3d_shape(self):
        state = tf.constant([[[20, 19], [18, 17]],
                             [[16, 15], [14, 13]],
                             [[12, 11], [10, 9]]], dtype=tf.float32)
        stoichiometry = tf.constant([[-1, 1, 0], [0, -1, 1]], dtype=tf.float32)
        def h(state):
            return tf.stack([[[0.1, 0.1], [0.1, 0.1]],
                             [[0.1, 0.1], [0.1, 0.1]]])
        propagate_fn = chain_binomial_propagate(h, 1, stoichiometry)
        new_state = propagate_fn(state)
        self.assertListEqual(list(new_state.shape), [3, 2, 2])

if __name__ == '__main__':
    unittest.main()
