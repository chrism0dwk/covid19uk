import unittest
import tensorflow as tf
import matplotlib.pyplot as plt
import time

from covid.impl.chainbinom_simulate import chain_binomial_simulate

class TestSimulator(unittest.TestCase):

    def test_homogenousSIR(self):
        state = tf.constant([[999, 1500], [1, 10], [0, 0]], dtype=tf.float64)

        def h(param):
            def f(state):
                rates = [param[0] * state[1]/tf.reduce_sum(state, axis=0),
                         tf.broadcast_to([param[1]], shape=[state.shape[1]])]
                indices = [[0, 1], [1, 2]]
                rate_matrix = tf.scatter_nd(updates=rates,
                                            indices=indices,
                                            shape=[state.shape[0], state.shape[0], state.shape[1]])
                return rate_matrix
            return f

        @tf.function
        def simulate_one(param):
            """One realisation of the epidemic process
            :param param: 1d tensor of parameters beta and gamma"""
            t, sim = chain_binomial_simulate(h(param), state, 0., 365., 1.)
            return sim

        @tf.function
        def simulate(param):
            """Simulate a bunch of epidemics at once"""
            sim = tf.map_fn(simulate_one, param, parallel_iterations=10)
            return sim

        start = time.perf_counter()
        param = tf.broadcast_to(tf.constant([0.4, 0.14], dtype=tf.float64), shape=[100, 2])
        sim = simulate(param)
        end = time.perf_counter()

        print(f"Complete in {end-start} seconds")
        plt.plot(sim[0, :, 1, :])
        plt.show()


if __name__ == '__main__':
    unittest.main()
