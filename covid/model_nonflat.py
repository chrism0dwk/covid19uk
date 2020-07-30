import collections
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

class SEIRState(collections.namedtuple(
    'SEIRState',
    ['S', 'E', 'I', 'R'])):
  pass

def power_iteration(ngt, tol=1e-3):
    ngt1, ngt2 = ngt   # [L, A, A],  [L, A, L]
    dtype = dtype_util.common_dtype([ngt1, ngt2], dtype_hint=np.float64)

    nlads, nages = ngt1.shape[0:2]
    b = tf.random.normal([nlads, nages], dtype=dtype)
    epsilon = tf.constant(1., dtype=dtype)

    cond = lambda epsilon, *_: tf.reduce_max(epsilon) > tol
    def body(epsilon, i, b):
        ngtb = (tf.einsum('...ijk,...lk->...ij', ngt1, b) +  # [L, A, A] @ [L, A]
                tf.einsum('...ijk,...kl->...ij', ngt2, b))   # ([A, A] @ [L, A]^T)^T
        norm = tf.sqrt(tf.reduce_sum(ngtb**2, axis=[-2, -1]))
        b_new = ngtb / norm
        epsilon = tf.reduce_sum(tf.math.squared_difference(b_new, b), axis=[-2, -1])
        return epsilon, i + 1, b_new

    eps, i, b = tf.while_loop(cond, body, [epsilon, 0, b])
    return b, i


def rayleigh_quotient(ngt, b):
    ngt1, ngt2 = ngt
    #b = tf.expand_dims(b, -1)  #tf.reshape(b, [b.shape[0], 1])

    ngtb = (tf.einsum('...ijk,...lk->...ij', ngt1, b) +  # [L, A, A] @ [L, A]
            tf.einsum('...ijk,...kl->...ij', ngt2, b))   # ([A, A] @ [L, A]^T)^T
    numerator = tf.einsum('...ij,...ij->...', b, ngtb)
    denominator = tf.einsum('...ij,...ij->...', b, b)
    print(numerator, denominator)
    return numerator / denominator


def dense_to_block_diagonal(A, n_blocks):
    A_dense = tf.linalg.LinearOperatorFullMatrix(A)
    eye = tf.linalg.LinearOperatorIdentity(n_blocks, dtype=A.dtype)
    A_block = tf.linalg.LinearOperatorKronecker([eye, A_dense])
    return A_block


def load_data(paths, settings, dtype):
    M_tt, age_groups = load_age_mixing(paths['age_mixing_matrix_term'])
    M_hh, _ = load_age_mixing(paths['age_mixing_matrix_hol'])

    C = collapse_commute_data(paths['mobility_matrix'])
    la_names = C.index.to_numpy()

    w_period = [settings['inference_period'][0], settings['prediction_period'][1]]
    W = load_commute_volume(paths['commute_volume'], w_period)['percent']

    pop = collapse_pop(paths['population_size'])

    M_tt = M_tt.astype(DTYPE)
    M_hh = M_hh.astype(DTYPE)
    C = C.to_numpy().astype(DTYPE)
    np.fill_diagonal(C, 0.)
    W = W.astype(DTYPE)
    pop['n'] = pop['n'].astype(DTYPE)

    return {'M_tt': M_tt, 'M_hh': M_hh,
            'C': C, 'la_names': la_names,
            'age_groups': age_groups,
            'W': W, 'pop': pop}


class CovidUK:
    def __init__(self,
                 age_mixing_matrix_term: np.float64,
                 age_mixing_matrix_hol: np.float64,
                 commute_volume: np.float64,
                 commute_matrix: np.float64,
                 population_matrix: np.float64,

                 date_range: list,
                 holidays: list,
                 lockdown: list,
                 time_step: np.int64):
        """Represents a CovidUK ODE model

        :param date_range: a time range [start, end)
        :param holidays: a list of length-2 tuples containing dates of holidays
        :param lockdown: a length-2 tuple of start and end of lockdown measures
        :param time_step: a time step to use in the discrete time simulation
        """
        dtype = dtype_util.common_dtype(
            [age_mixing_matrix_term, age_mixing_matrix_hol,
             commute_volume, commute_matrix, population_matrix],
            dtype_hint=np.float64)

        self.age_mixing_matrix_term = tf.convert_to_tensor(
            age_mixing_matrix_term, dtype=dtype)
        self.age_mixing_matrix_hol = tf.convert_to_tensor(
            age_mixing_matrix_hol, dtype=dtype)

        # Create one linear operator comprising both the term and holiday
        # matrices. This is nice because
        #   - the dense "M" parts will take up memory of shape [2, M, M]
        #   - the identity matrix will only take up memory of shape [M]
        #   - matmuls/matvecs will be quite efficient because of the
        #     LinearOperatorKronecker structure and diagonal structure of the
        #     identity piece thereof.
        # It should be sufficiently efficient that we can just get rid of the
        # control flow switching between the two operators, and instead just do
        # both matmuls in one big (vectorized!) pass, and pull out what we want
        # after the fact with tf.gather.
        self.age_mixing_matrices = np.stack([
            age_mixing_matrix_term, age_mixing_matrix_hol], axis=0)

        self.average_mixing_across_ages = tf.reduce_mean(age_mixing_matrix_term)

        # Number of commutes from work to home. Symmetrizing assumes
        # everyone goes to work and back again.
        self.commute_matrix = commute_matrix + tf.transpose(commute_matrix)
        self.commute_volume = tf.constant(commute_volume, dtype=dtype)

        self.population_matrix = tf.constant(population_matrix, dtype=dtype)
        # keepdims so that the result has shape [n_lads, 1], for friendly
        # broadcasting divides later on.
        self.lad_pops = tf.reduce_sum(
            self.population_matrix, keepdims=True, axis=-1)

        self.time_step = time_step
        one_step = np.timedelta64(int(time_step), 'D')
        self.times = np.arange(date_range[0], date_range[1] + one_step, one_step)

        self.m_select = np.int64((self.times >= holidays[0]) &
                                 (self.times < holidays[1]))
        self.lockdown_select = np.int64((self.times >= lockdown[0]) &
                                        (self.times < lockdown[1]))
        self.max_t = self.m_select.shape[0] - 1

class CovidUKODE(CovidUK):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.solver = tode.DormandPrince()


    def make_h(self, param):

        def h_fn(t, state):
            S, E, I, R = state
            # Integrator may produce time values outside the range desired, so
            # we clip, implicitly assuming the outside dates have the same
            # holiday status as their nearest neighbors in the desired range.
            t_idx = tf.clip_by_value(tf.cast(t, tf.int64), 0, self.max_t)
            m_switch = tf.gather(self.m_select, t_idx)
            lockdown = tf.gather(self.lockdown_select, t_idx)
            beta = tf.where(lockdown == 0, param['beta1'], param['beta1']*param['beta3'])

            infec_rate = beta * (
                tf.gather(tf.linalg.einsum('...ij,...kj->...ki',
                                           self.age_mixing_matrices, I),
                          m_switch) +
                param['beta2'] *
                self.average_mixing_across_ages *
                tf.gather(self.commute_volume, t_idx) ** param['omega'] *
                tf.linalg.einsum('...ij,...jk->...ik',
                                 self.commute_matrix,
                                 I / self.lad_pops))

            new_infections = S * infec_rate / self.population_matrix

            dS = -new_infections
            dE = new_infections - param['nu'] * E
            dI = param['nu'] * E - param['gamma'] * I
            dR = param['gamma'] * I

            return SEIRState(dS, dE, dI, dR)

        return h_fn

    def simulate(self, param, state_init, solver_state=None):
        h = self.make_h(param)
        t = np.arange(self.times.shape[0])
        results = self.solver.solve(ode_fn=h, initial_time=t[0], initial_state=state_init,
                                    solution_times=t, previous_solver_internal_state=solver_state)
        return results.times, results.states, results.solver_internal_state


    def ngt(self, param, t, S):
        t_idx = tf.clip_by_value(tf.cast(t, tf.int64), 0, self.max_t)
        m_switch = tf.gather(self.m_select, t_idx)
        commute_volume = tf.pow(tf.gather(self.commute_volume, t_idx), param['omega'])
        lockdown = tf.gather(self.lockdown_select, t_idx)
        beta = tf.where(lockdown == 0, param['beta1'], param['beta1'] * param['beta3'])

        commute = (param['beta2'] *
                   self.average_mixing_across_ages *
                   commute_volume *
                   self.commute_matrix / self.lad_pops)

        current_age_mixing = tf.gather(self.age_mixing_matrices, m_switch)

        ngt1 = (beta[..., None, None] *
                (S / self.population_matrix)[..., None, :] *
                current_age_mixing *
                param['gamma'])
        # => [L, A, A]
        ngt2 = (beta[..., None, None] *
                commute[:, None, :] *
                (S / self.population_matrix)[..., :, None] *
                param['gamma'])
        # => [L, A, L]

        return ngt1, ngt2

    def eval_Rt(self, param, t, S):
        ngt = self.ngt(param, t, S)
        dom_eigen_vec, i = power_iteration(ngt, tol=tf.cast(1e-8, tf.float64))
        Rt = rayleigh_quotient(ngt, dom_eigen_vec)
        return Rt


def covid19uk_logp(y, sim, phi, r):
    """log_probability function for case data
    :param y: a full time/lad/age dataset of case data
    :param sim: a simulation
    :param phi: the case detection rate
    :param r: negative binomial overdispersion parameter
    """
    # Daily increments in removed
    r_incr = sim[1:, :, 3] - sim[:-1, :, 3]
    r_incr = tf.reduce_sum(r_incr, axis=1)
    lambda_ = tf.reshape(r_incr, [-1]) * phi
    y = y.sum(level=0).to_numpy()
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
            commute_volume = tf.pow(tf.gather(self.commute_volume, t_idx), param['omega'])

            infec_rate = param['beta1'] * (
                tf.gather(self.age_mixing_matrices.matvec(state[:, 2]), m_switch) +
                param['beta2'] * self.average_mixing_across_ages * commute_volume * self.commute_matrix.matvec(state[:, 2] / self.lad_pops))
            infec_rate = infec_rate / self.population_matrix

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
