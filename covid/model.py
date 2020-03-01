"""Functions for infection rates"""

import tensorflow as tf

from covid.impl.chainbinom_simulate import chain_binomial_simulate

class CovidUK:
    def __init__(self, K, T, W):
        self.K = K
        self.T = T
        self.W = W

        self.stoichiometry = [[-1, 1, 0, 0],
                              [0, -1, 1, 0],
                              [0, 0, -1, 1]]

    @tf.function
    def h(self, state):
        state = tf.unstack(state, axis=0)
        S, E, I, R = state

        hazard_rates = tf.stack([
            self.param['beta1'] * tf.dot(self.T, tf.dot(self.K, I)),
            self.param['nu'],
            self.param['gamma']
        ])
        return hazard_rates

    @tf.function
    def sample(self, initial_state, time_lims, param):
        self.param = param
        return chain_binomial_simulate(self.h, initial_state, time_lims[0],
                                       time_lims[1], 1., self.stoichiometry)


class Homogeneous:
    def __init__(self):

        self.stoichiometry = tf.constant([[-1, 1, 0, 0],
                                         [0, -1, 1, 0],
                                         [0, 0, -1, 1]], dtype=tf.float32)

    def h(self, state):
        state = tf.unstack(state, axis=0)
        S, E, I, R = state

        hazard_rates = tf.stack([
            self.param['beta'] * I/tf.reduce_sum(state),
            self.param['nu'] * tf.ones_like(I),
            self.param['gamma'] * tf.ones_like(I)
        ])
        return hazard_rates

    @tf.function
    def sample(self, initial_state, time_lims, param):
        self.param = param
        return chain_binomial_simulate(self.h, initial_state, time_lims[0],
                                       time_lims[1], 1., self.stoichiometry)
