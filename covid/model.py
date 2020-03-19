"""Functions for infection rates"""
from warnings import warn
import tensorflow as tf
import tensorflow_probability as tfp

tode = tfp.math.ode
import numpy as np

from covid.impl.chainbinom_simulate import chain_binomial_simulate

def power_iteration(A, tol=1e-3):
    b_k = tf.random.normal([A.shape[1], 1])
    epsilon = 1.
    i = 0
    while tf.greater(epsilon, tol):
        b_k1 = tf.matmul(A, b_k)
        b_k1_norm = tf.linalg.norm(b_k1)
        b_k_new = b_k1 / b_k1_norm
        epsilon = tf.reduce_sum(tf.pow(b_k_new-b_k, 2))
        b_k = b_k_new
        i += 1
    return b_k, i

#@tf.function
def rayleigh_quotient(A, b):
    b = tf.reshape(b, [b.shape[0], 1])
    numerator = tf.matmul(tf.transpose(b), tf.matmul(A, b))
    denominator = tf.matmul(tf.transpose(b), b)
    return numerator / denominator

class CovidUK:
    def __init__(self, K, T, W):
        self.K = K
        self.T = T
        self.W = W

        self.stoichiometry = [[-1, 1, 0, 0],
                              [0, -1, 1, 0],
                              [0, 0, -1, 1]]

    def h(self, state):
        state = tf.unstack(state, axis=0)
        S, E, I, R = state

        hazard_rates = tf.stack([
            self.param['beta1'] * tf.dot(self.T, tf.dot(self.K, I))/self.K.shape[0],
            self.param['nu'],
            self.param['gamma']
        ])
        return hazard_rates

    #@tf.function
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


def dense_to_block_diagonal(A, n_blocks):
    A_dense = tf.linalg.LinearOperatorFullMatrix(A)
    eye = tf.linalg.LinearOperatorIdentity(n_blocks, dtype=tf.float64)
    A_block = tf.linalg.LinearOperatorKronecker([eye, A_dense])
    return A_block


class CovidUKODE:  # TODO: add background case importation rate to the UK, e.g. \epsilon term.
    def __init__(self, M_tt, M_hh, C, N, start, end, holidays, bg_max_t, t_step):
        """Represents a CovidUK ODE model

        :param K_tt: a MxM matrix of age group mixing in term time
        :param K_hh: a MxM matrix of age group mixing in holiday time
        :param holidays: a list of length-2 tuples containing dates of holidays
        :param C: a n_ladsxn_lads matrix of inter-LAD commuting
        :param N: a vector of population sizes in each LAD
        :param n_lads: the number of LADS
        """
        self.n_ages = M_tt.shape[0]
        self.n_lads = C.shape[0]

        self.Kbar = tf.reduce_mean(tf.cast(M_tt, tf.float64))
        self.M = tf.tuple([dense_to_block_diagonal(tf.cast(M_tt, tf.float64), self.n_lads),
                           dense_to_block_diagonal(tf.cast(M_hh, tf.float64), self.n_lads)])

        C = tf.cast(C, tf.float64)
        self.C = tf.linalg.LinearOperatorFullMatrix(C + tf.transpose(C))
        shp = tf.linalg.LinearOperatorFullMatrix(np.ones_like(M_tt, dtype=np.float64))
        self.C = tf.linalg.LinearOperatorKronecker([self.C, shp])

        self.N = tf.constant(N, dtype=tf.float64)
        N_matrix = tf.reshape(self.N, [self.n_lads, self.n_ages])
        N_sum = tf.reduce_sum(N_matrix, axis=1)
        N_sum = N_sum[:, None] * tf.ones([1, self.n_ages], dtype=tf.float64)
        self.N_sum = tf.reshape(N_sum, [-1])

        self.times = np.arange(start, end, np.timedelta64(t_step, 'D'))
        m_select = (np.less_equal(holidays[0], self.times) & np.less(self.times, holidays[1])).astype(np.int64)
        self.m_select = tf.constant(m_select, dtype=tf.int64)
        self.bg_select = tf.constant(np.less(self.times, bg_max_t), dtype=tf.int64)
        self.solver = tode.DormandPrince()

    def make_h(self, param):

        def h_fn(t, state):
            state = tf.unstack(state, axis=0)
            S, E, I, R = state
            t = tf.clip_by_value(tf.cast(t, tf.int64), 0, self.m_select.shape[0]-1)
            m_switch = tf.gather(self.m_select, t)
            epsilon = param['epsilon'] * tf.cast(tf.gather(self.bg_select, t), tf.float64)
            if m_switch == 0:
               infec_rate = param['beta1'] * tf.linalg.matvec(self.M[0], I)
            else:
                infec_rate = param['beta1'] * tf.linalg.matvec(self.M[1], I)
            infec_rate += param['beta1'] * param['beta2'] * self.Kbar * tf.linalg.matvec(self.C, I / self.N_sum)
            infec_rate = S / self.N * (infec_rate + epsilon)

            dS = -infec_rate
            dE = infec_rate - param['nu'] * E
            dI = param['nu'] * E - param['gamma'] * I
            dR = param['gamma'] * I

            df = tf.stack([dS, dE, dI, dR])
            return df

        return h_fn

    def create_initial_state(self, init_matrix=None):
        if init_matrix is None:
            I = np.zeros(self.N.shape, dtype=np.float64)
            I[149*17+10] = 30. # Middle-aged in Surrey
        else:
            np.testing.assert_array_equal(init_matrix.shape, [self.n_lads, self.n_ages],
                                          err_msg=f"init_matrix does not have shape [<num lads>,<num ages>] \
                                          ({self.n_lads},{self.n_ages})")
            I = init_matrix.flatten()
        S = self.N - I
        E = np.zeros(self.N.shape, dtype=np.float64)
        R = np.zeros(self.N.shape, dtype=np.float64)
        return np.stack([S, E, I, R])

    @tf.function
    def simulate(self, param, state_init, solver_state=None):
        h = self.make_h(param)
        t = np.arange(self.times.shape[0])
        results = self.solver.solve(ode_fn=h, initial_time=t[0], initial_state=state_init,
                                    solution_times=t, previous_solver_internal_state=solver_state)
        return results.times, results.states, results.solver_internal_state

    def ngm(self, param):
        infec_rate = param['beta1'] * self.M[0].to_dense()
        infec_rate += param['beta1'] * param['beta2'] * self.Kbar * self.C.to_dense() / self.N_sum[None, :]
        ngm = infec_rate / param['gamma']
        return ngm

    def eval_R0(self, param, tol=1e-8):
        ngm = self.ngm(param)
        # Dominant eigen value by power iteration
        dom_eigen_vec, i = power_iteration(ngm, tol=tol)
        R0 = rayleigh_quotient(ngm, dom_eigen_vec)
        return tf.squeeze(R0), i


def covid19uk_logp(y, sim, phi):
        # Sum daily increments in removed
        r_incr = sim[1:, 3, :] - sim[:-1, 3, :]
        r_incr = tf.reduce_sum(r_incr, axis=1)
        y_ = tfp.distributions.Poisson(rate=phi*r_incr)
        return y_.log_prob(y)
