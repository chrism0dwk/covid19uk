"""Functions for infection rates"""

import tensorflow as tf
import tensorflow_probability as tfp
tode = tfp.math.ode
import numpy as np

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


class CovidUKODE:

    def __init__(self, K, T, N, n_lads=152):
        """Represents a CovidUK ODE model

        :param K: a MxM matrix of age group mixing
        :param T: a n_ladsxn_lads matrix of inter-LAD connectivity
        :param N: a vector of population sizes in each LAD
        :param n_lads: the number of LADS
        """
        K = tf.linalg.LinearOperatorFullMatrix(K)
        eye = tf.linalg.LinearOperatorDiag(n_lads)
        self.K = tf.linalg.LinearOperatorKronecker([K, eye])

        T = tf.linalg.LinearOperatorFullMatrix(T)
        shp = tf.linalg.LinearOperatorFullMatrix(np.ones([n_lads, n_lads]))
        self.T = tf.linalg.LinearOperatorKronecker([T, shp])

    def make_h(self, param):
        @tf.function
        def h_fn(state):
            state = tf.unstack(state, axis=0)
            S, E, I, R = state

            infec_rate = tf.matvec(self.K, I)
            infec_rate += tf.reduce_mean(self.K) * tf.matvec(self.T, I/self.N)
            infec_rate  = infec_rate * S / self.N

            dS = -infec_rate
            dE = infec_rate - param['nu'] * E
            dI = param['nu'] * E - param['gamma'] * I
            dR = param['gamma'] * I

            df = tf.stack([dS, dE, dI, dR])
            return df
        return h_fn

    @tf.function
    def simulate(self, param, state_init, t_start, t_end, t_step=1.):
        h = self.make_h(param)
        t = np.linspace(t_start, t_end, t_step)
        sim = tode.DormandPrince(first_step_size=1., max_num_steps=5000).solve(h, t_start, state_init,
                                                                               solution_times=t)
        return sim