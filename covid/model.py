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
            self.param['beta'] * I / tf.reduce_sum(state),
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

    def __init__(self, K, T, N, n_lads):
        """Represents a CovidUK ODE model

        :param K: a MxM matrix of age group mixing
        :param T: a n_ladsxn_lads matrix of inter-LAD connectivity
        :param N: a vector of population sizes in each LAD
        :param n_lads: the number of LADS
        """
        self.Kbar = tf.reduce_mean(K)
        self.K = tf.linalg.LinearOperatorFullMatrix(K)
        eye = tf.linalg.LinearOperatorIdentity(n_lads)
        self.K = tf.linalg.LinearOperatorKronecker([self.K, eye])

        self.T = tf.linalg.LinearOperatorFullMatrix(T)
        shp = tf.linalg.LinearOperatorFullMatrix(np.ones_like(K, dtype=np.float32))
        self.T = tf.linalg.LinearOperatorKronecker([self.T, shp])

        self.N = N
        self.n_lads = n_lads

    def make_h(self, param):
        def h_fn(t, state):
            state = tf.unstack(state, axis=0)
            S, E, I, R = state

            infec_rate = tf.linalg.matvec(self.K, I)
            infec_rate += self.Kbar * tf.linalg.matvec(self.T, I / self.N)
            infec_rate = param['beta'] * S / self.N * infec_rate

            dS = -infec_rate
            dE = infec_rate - param['nu'] * E
            dI = param['nu'] * E - param['gamma'] * I
            dR = param['gamma'] * I

            df = tf.stack([dS, dE, dI, dR])
            return df

        return h_fn

    def create_initial_state(self, init_matrix=None):
        if init_matrix is None:
            I = np.ones(self.N.shape, dtype=np.float32)
            S = self.N - I
            E = np.zeros(self.N.shape, dtype=np.float32)
            R = np.zeros(self.N.shape, dtype=np.float32)
            return np.stack([S, E, I, R])

    @tf.function
    def simulate(self, param, state_init, t_start, t_end, t_step=1.):
        h = self.make_h(param)
        t = np.arange(start=t_start, stop=t_end, step=t_step)
        print(f"Running simulation with times {t_start}-{t_end}:{t_step}")
        solver = tode.DormandPrince()
        results = solver.solve(ode_fn=h, initial_time=t_start, initial_state=state_init, solution_times=t)
        return results.times, results.states
