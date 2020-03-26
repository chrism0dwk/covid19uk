"""Functions for infection rates"""
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import dtype_util
import numpy as np

from covid.impl.chainbinom_simulate import chain_binomial_simulate

tode = tfp.math.ode
tla = tf.linalg


def power_iteration(A, tol=1e-3):
    b_k = tf.random.normal([A.shape[1], 1], dtype=A.dtype)
    epsilon = tf.constant(1., dtype=A.dtype)
    i = 0
    while tf.greater(epsilon, tol):
        b_k1 = tf.matmul(A, b_k)
        b_k1_norm = tf.linalg.norm(b_k1)
        b_k_new = b_k1 / b_k1_norm
        epsilon = tf.reduce_sum(tf.pow(b_k_new-b_k, 2))
        b_k = b_k_new
        i += 1
    return b_k, i


def rayleigh_quotient(A, b):
    b = tf.reshape(b, [b.shape[0], 1])
    numerator = tf.matmul(tf.transpose(b), tf.matmul(A, b))
    denominator = tf.matmul(tf.transpose(b), b)
    return numerator / denominator


def dense_to_block_diagonal(A, n_blocks):
    A_dense = tf.linalg.LinearOperatorFullMatrix(A)
    eye = tf.linalg.LinearOperatorIdentity(n_blocks, dtype=A.dtype)
    A_block = tf.linalg.LinearOperatorKronecker([eye, A_dense])
    return A_block


class CovidUKODE:  # TODO: add background case importation rate to the UK, e.g. \epsilon term.
    def __init__(self,
                 M_tt: np.float64,
                 M_hh: np.float64,
                 W: np.float64,
                 C: np.float64,
                 N: np.float64,
                 date_range: list,
                 holidays: list,
                 t_step: np.int64):
        """Represents a CovidUK ODE model

        :param K_tt: a MxM matrix of age group mixing in term time
        :param K_hh: a MxM matrix of age group mixing in holiday time
        :param W: Commuting volume
        :param date_range: a time range [start, end)
        :param holidays: a list of length-2 tuples containing dates of holidays
        :param C: a n_ladsxn_lads matrix of inter-LAD commuting
        :param N: a vector of population sizes in each LAD
        :param n_lads: the number of LADS
        """
        dtype = dtype_util.common_dtype([M_tt, M_hh, W, C, N], dtype_hint=np.float64)
        self.n_ages = M_tt.shape[0]
        self.n_lads = C.shape[0]
        self.M_tt = tf.convert_to_tensor(M_tt, dtype=tf.float64)
        self.M_hh = tf.convert_to_tensor(M_hh, dtype=tf.float64)

        # Create one linear operator comprising both the term and holiday
        # matrices. This is nice because
        #   - the dense "M" parts will take up memory of shape [2, M, M]
        #   - the identity matirix will only take up memory of shape [M]
        #   - matmuls/matvecs will be quite efficient because of the
        #     LinearOperatorKronecker structure and diagonal structure of the
        #     identity piece thereof.
        # It should be sufficiently efficient that we can just get rid of the
        # control flow switching between the two operators, and instead just do
        # both matmuls in one big (vectorized!) pass, and pull out what we want
        # after the fact with tf.gather.
        self.M = dense_to_block_diagonal(
            np.stack([M_tt, M_hh], axis=0), self.n_lads)

        self.Kbar = tf.reduce_mean(M_tt)

        self.C = tf.linalg.LinearOperatorFullMatrix(C + tf.transpose(C))
        shp = tf.linalg.LinearOperatorFullMatrix(tf.ones_like(M_tt, dtype=dtype))
        self.C = tf.linalg.LinearOperatorKronecker([self.C, shp])
        self.W = tf.constant(W, dtype=dtype)
        self.N = tf.constant(N, dtype=dtype)
        N_matrix = tf.reshape(self.N, [self.n_lads, self.n_ages])
        N_sum = tf.reduce_sum(N_matrix, axis=1)
        N_sum = N_sum[:, None] * tf.ones([1, self.n_ages], dtype=dtype)
        self.N_sum = tf.reshape(N_sum, [-1])

        self.times = np.arange(date_range[0], date_range[1], np.timedelta64(t_step, 'D'))

        self.m_select = np.int64((self.times >= holidays[0]) &
                                 (self.times < holidays[1]))
        self.max_t = self.m_select.shape[0] - 1

        self.solver = tode.DormandPrince()

    def make_h(self, param):

        def h_fn(t, state):

            S, E, I, R = state
            # Integrator may produce time values outside the range desired, so
            # we clip, implicitly assuming the outside dates have the same
            # holiday status as their nearest neighbors in the desired range.
            t_idx = tf.clip_by_value(tf.cast(t, tf.int64), 0, self.max_t)
            m_switch = tf.gather(self.m_select, t_idx)
            commute_volume = tf.pow(tf.gather(self.W, t_idx), param['omega'])

            infec_rate = param['beta1'] * (
                tf.gather(self.M.matvec(I), m_switch) +
                param['beta2'] * self.Kbar * commute_volume * self.C.matvec(I / self.N_sum))
            infec_rate = infec_rate / self.N

            SE = infec_rate
            EI = param['nu']
            IR = param['gamma']

            p = 1 - tf.exp([SE, EI, IR])
            return p

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

    def simulate(self, param, state_init, solver_state=None):
        h = self.make_h(param)
        t = np.arange(self.times.shape[0])
        results = self.solver.solve(ode_fn=h, initial_time=t[0], initial_state=state_init,
                                    solution_times=t, previous_solver_internal_state=solver_state)
        return results.times, results.states, results.solver_internal_state

    def ngm(self, param):
        infec_rate = param['beta1'] * (
            self.M.to_dense()[0, ...] +
            (param['beta2'] * self.Kbar * self.C.to_dense() /
             self.N_sum[np.newaxis, :]))
        ngm = infec_rate / param['gamma']
        return ngm

    def eval_R0(self, param, tol=1e-8):
        ngm = self.ngm(param)
        # Dominant eigen value by power iteration
        dom_eigen_vec, i = power_iteration(ngm, tol=tf.cast(tol, tf.float64))
        R0 = rayleigh_quotient(ngm, dom_eigen_vec)
        return tf.squeeze(R0), i


def covid19uk_logp(y, sim, phi):
        # Sum daily increments in removed
        r_incr = sim[1:, 3, :] - sim[:-1, 3, :]
        r_incr = tf.reduce_sum(r_incr, axis=1)
        y_ = tfp.distributions.Poisson(rate=phi*r_incr)
        return y_.log_prob(y)
