import unittest
import tensorflow as tf

from covid.impl.chainbinom_simulate import chain_binomial_simulate

class TestSimulator(unittest.TestCase):

    def test_homogenousSIR(self):
        state = tf.constant([[999], [1], [0]], dtype=tf.float32)
        stoichiometry = tf.constant([[-1, 1, 0],
                                     [0, -1, 1]], dtype=tf.float32)
        def h(state):
            return tf.stack([[0.4 * state[1, 0]/tf.reduce_sum(state)],
                             [0.14]])
        sim = chain_binomial_simulate(h, state, 0., 365., 1, stoichiometry)
        print(sim)


if __name__ == '__main__':
    unittest.main()
