"""Functions for infection rates"""
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import dtype_util
import numpy as np

from covid.rdata import load_mobility_matrix, load_population, load_age_mixing
from covid.pydata import load_commute_volume, collapse_commute_data, collapse_pop
from covid.impl.chainbinom_simulate import chain_binomial_simulate

tode = tfp.math.ode
tla = tf.linalg

DTYPE = np.float64


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


def load_data(paths, settings, dtype=DTYPE):
    M_tt, age_groups = load_age_mixing(paths['age_mixing_matrix_term'])
    M_hh, _ = load_age_mixing(paths['age_mixing_matrix_hol'])

    C = collapse_commute_data(paths['mobility_matrix'])
    la_names = C.index.to_numpy()
    C = C.to_numpy()
    np.fill_diagonal(C, 0.)

    w_period = [settings['inference_period'][0], settings['prediction_period'][1]]
    W = load_commute_volume(paths['commute_volume'], w_period)['percent']

    pop = collapse_pop(paths['population_size'])

    M_tt = M_tt.astype(DTYPE)
    M_hh = M_hh.astype(DTYPE)
    C = C.astype(DTYPE)
    W = W.astype(DTYPE)
    pop['n'] = pop['n'].astype(DTYPE)

    return {'M_tt': M_tt, 'M_hh': M_hh,
            'C': C, 'la_names': la_names,
            'age_groups': age_groups,
            'W': W, 'pop': pop}


class CovidUK:
    def __init__(self,
                 M_tt: np.float64,
                 M_hh: np.float64,
                 W: np.float64,
                 C: np.float64,
                 N: np.float64,
                 date_range: list,
                 holidays: list,
                 lockdown: list,
                 time_step: np.int64):
        """Represents a CovidUK ODE model

        :param M_tt: a MxM matrix of age group mixing in term time
        :param M_hh: a MxM matrix of age group mixing in holiday time
        :param W: Commuting volume
        :param C: a n_ladsxn_lads matrix of inter-LAD commuting
        :param N: a vector of population sizes in each LAD
        :param date_range: a time range [start, end)
        :param holidays: a list of length-2 tuples containing dates of holidays
        :param lockdown: a length-2 tuple of start and end of lockdown measures
        :param time_step: a time step to use in the discrete time simulation
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

        self.time_step = time_step
        one_step = np.timedelta64(int(time_step), 'D')
        self.times = np.arange(date_range[0], date_range[1] + one_step, one_step)

        self.m_select = np.int64((self.times >= holidays[0]) &
                                 (self.times < holidays[1]))
        self.lockdown_select = np.int64((self.times >= lockdown[0]) &
                                        (self.times < lockdown[1]))
        self.max_t = self.m_select.shape[0] - 1

    def create_initial_state(self, init_matrix=None):
        if init_matrix is None:
            I = np.zeros(self.N.shape, dtype=DTYPE)
            I[149*17+10] = 30.  # Middle-aged in Surrey
        else:
            I = tf.convert_to_tensor(init_matrix, dtype=DTYPE)
        S = self.N - I
        E = tf.zeros(self.N.shape, dtype=DTYPE)
        R = tf.zeros(self.N.shape, dtype=DTYPE)
        return tf.stack([S, E, I, R], axis=-1)


class CovidUKODE(CovidUK):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.solver = tode.DormandPrince()


    def make_h(self, param):

        def h_fn(t, state):

            S, E, I, R = tf.unstack(state, axis=-1)
            # Integrator may produce time values outside the range desired, so
            # we clip, implicitly assuming the outside dates have the same
            # holiday status as their nearest neighbors in the desired range.
            t_idx = tf.clip_by_value(tf.cast(t, tf.int64), 0, self.max_t)
            m_switch = tf.gather(self.m_select, t_idx)
            commute_volume = tf.pow(tf.gather(self.W, t_idx), param['omega'])
            lockdown = tf.gather(self.lockdown_select, t_idx)
            beta = tf.where(lockdown == 0, param['beta1'], param['beta1']*param['beta3'])

            infec_rate = beta * (
                tf.gather(self.M.matvec(I), m_switch) +
                param['beta2'] * self.Kbar * commute_volume * self.C.matvec(I / self.N_sum))
            infec_rate = S * infec_rate / self.N

            dS = -infec_rate
            dE = infec_rate - param['nu'] * E
            dI = param['nu'] * E - param['gamma'] * I
            dR = param['gamma'] * I

            df = tf.stack([dS, dE, dI, dR], axis=-1)
            return df

        return h_fn

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

def covid19uk_logp(y, sim, phi, r):
    """log_probability function for case data
    :param y: a full time/lad/age dataset of case data
    :param sim: a simulation
    :param phi: the case detection rate
    :param r: negative binomial overdispersion parameter
    """
    # Daily increments in removed
    r_incr = sim[1:, :, 3] - sim[:-1, :, 3]
    # Sum
    #r_incr = tf.reshape(r_incr, [r_incr.shape[0]] + [149, 17])
    r_incr = tf.reduce_sum(r_incr, axis=1)
    lambda_ = tf.reshape(r_incr, [-1]) * phi
    y = y.sum(level=0)
    y_ = tfp.distributions.NegativeBinomial(r, probs=lambda_/(r+lambda_))
    return tf.reduce_sum(y_.log_prob(y))


class CovidUKStochastic(CovidUK):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def make_h(self, param):
        """Constructs a function that takes `state` and outputs a
        transition rate matrix (with 0 diagonal).
        """

        def h(t, state):
            """Computes a transition rate matrix

            :param state: a tensor of shape [ns, nc] for ns states and nc population strata. States
              are S, E, I, R.  We arrange the state like this because the state vectors are then arranged
              contiguously in memory for fast calculation below.
            :return a tensor of shape [ns, ns, nc] containing transition matric for each i=0,...,(c-1)
            """
            t_idx = tf.clip_by_value(tf.cast(t, tf.int64), 0, self.max_t)
            m_switch = tf.gather(self.m_select, t_idx)
            commute_volume = tf.pow(tf.gather(self.W, t_idx), param['omega'])

            infec_rate = param['beta1'] * (
                tf.gather(self.M.matvec(state[:, 2]), m_switch) +
                param['beta2'] * self.Kbar * commute_volume * self.C.matvec(state[:, 2] / self.N_sum))
            infec_rate = infec_rate / self.N

            ei = tf.broadcast_to([param['nu']], shape=[state.shape[0]])
            ir = tf.broadcast_to([param['gamma']], shape=[state.shape[0]])

            # Scatter rates into a [ns, ns, nc] tensor
            n = state.shape[0]
            b = tf.stack([tf.range(n),
                          tf.zeros(n, dtype=tf.int32),
                          tf.ones(n, dtype=tf.int32)], axis=-1)
            indices = tf.stack([b, b + [0, 1, 1], b + [0, 2, 2]], axis=-2)
            # Un-normalised rate matrix (diag is 0 here)
            rate_matrix = tf.scatter_nd(indices=indices,
                                        updates=tf.stack([infec_rate, ei, ir], axis=-1),
                                        shape=[state.shape[0],
                                               state.shape[1],
                                               state.shape[1]])
            return rate_matrix
        return h

    @tf.function(autograph=False, experimental_compile=True)
    def simulate(self, param, state_init):
        """Runs a simulation from the epidemic model

        :param param: a dictionary of model parameters
        :param state_init: the initial state
        :returns: a tuple of times and simulated states.
        """
        param = {k: tf.constant(v, dtype=tf.float64) for k, v in param.items()}
        hazard = self.make_h(param)
        t, sim = chain_binomial_simulate(hazard, state_init, np.float64(0.),
                                         np.float64(self.times.shape[0]), self.time_step)
        return t, sim
