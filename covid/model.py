"""Functions for infection rates"""
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import dtype_util
import numpy as np

from covid import config
from covid.impl.util import make_transition_matrix
from covid.rdata import load_age_mixing
from covid.pydata import load_commute_volume, load_mobility_matrix, load_population
from covid.impl.discrete_markov import (
    discrete_markov_simulation,
    discrete_markov_log_prob,
)

tla = tf.linalg

DTYPE = config.floatX


def power_iteration(A, tol=1e-3):
    b_k = tf.random.normal([A.shape[1], 1], dtype=A.dtype)
    epsilon = tf.constant(1.0, dtype=A.dtype)
    i = 0
    while tf.greater(epsilon, tol):
        b_k1 = tf.matmul(A, b_k)
        b_k1_norm = tf.linalg.norm(b_k1)
        b_k_new = b_k1 / b_k1_norm
        epsilon = tf.reduce_sum(tf.pow(b_k_new - b_k, 2))
        b_k = b_k_new
        i += 1
    return b_k, i


def rayleigh_quotient(A, b):
    b = tf.squeeze(b)
    numerator = tf.einsum("...i,...i->...", b, tf.linalg.matvec(A, b))
    denominator = tf.einsum("...i,...i->...", b, b)
    return numerator / denominator


def dense_to_block_diagonal(A, n_blocks):
    A_dense = tf.linalg.LinearOperatorFullMatrix(A)
    eye = tf.linalg.LinearOperatorIdentity(n_blocks, dtype=A.dtype)
    A_block = tf.linalg.LinearOperatorKronecker([eye, A_dense])
    return A_block


def load_data(paths, settings, dtype=DTYPE):
    M_tt, age_groups = load_age_mixing(paths["age_mixing_matrix_term"])
    M_hh, _ = load_age_mixing(paths["age_mixing_matrix_hol"])

    C = load_mobility_matrix(paths["mobility_matrix"])
    la_names = C.index.to_numpy()

    w_period = [settings["inference_period"][0], settings["inference_period"][1]]
    W = load_commute_volume(paths["commute_volume"], w_period)["percent"]

    pop = load_population(paths["population_size"])

    M_tt = M_tt.astype(DTYPE)
    M_hh = M_hh.astype(DTYPE)
    C = C.to_numpy().astype(DTYPE)
    np.fill_diagonal(C, 0.0)
    W = W.to_numpy().astype(DTYPE)
    pop = pop.to_numpy().astype(DTYPE)

    return {
        "M_tt": M_tt,
        "M_hh": M_hh,
        "C": C,
        "la_names": la_names,
        "age_groups": age_groups,
        "W": W,
        "pop": pop,
    }


class CovidUK:
    def __init__(
        self,
        W: np.float64,
        C: np.float64,
        N: np.float64,
        xi_freq: int,
        params: dict,
        initial_state: np.float64,
        initial_time: np.float64,
        time_step: np.int64,
        num_steps: np.int64,
    ):
        """Represents a CovidUK ODE model

        :param W: Commuting volume
        :param C: a n_ladsxn_lads matrix of inter-LAD commuting
        :param N: a vector of population sizes in each LAD
        :param date_range: a time range [start, end)
        :param beta_freq: the frequency at which beta changes
        :param time_step: a time step to use in the discrete time simulation
        """

        dtype = dtype_util.common_dtype([W, C, N, initial_state], dtype_hint=DTYPE)

        self.initial_state = tf.convert_to_tensor(initial_state, dtype=dtype)
        self.initial_time = initial_time
        self.n_lads = C.shape[0]

        C = tf.convert_to_tensor(C, dtype=dtype)
        self.C = C + tf.transpose(C)
        self.C = tf.linalg.set_diag(self.C, tf.zeros(self.C.shape[0], dtype=dtype))
        self.W = tf.constant(W, dtype=dtype)
        self.N = tf.constant(N, dtype=dtype)

        self.time_step = time_step
        self.num_steps = num_steps

        self.params = {
            k: tf.convert_to_tensor(v, dtype=dtype) for k, v in params.items()
        }

        self.xi_freq = np.int32(xi_freq)
        self.xi_select = np.arange(self.num_steps, dtype=np.int32) // self.xi_freq
        self.max_t = self.xi_select.shape[0] - 1

    def create_initial_state(self, init_matrix=None):
        I = tf.convert_to_tensor(init_matrix, dtype=DTYPE)
        S = self.N - I
        E = tf.zeros(self.N.shape, dtype=DTYPE)
        R = tf.zeros(self.N.shape, dtype=DTYPE)
        return tf.stack([S, E, I, R], axis=-1)


class CovidUKStochastic(CovidUK):

    stoichiometry = tf.constant(
        [[-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1]], dtype=DTYPE
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def make_h(self, param=None):
        """Constructs a function that takes `state` and outputs a
        transition rate matrix (with 0 diagonal).
        """

        if param is None:
            param = self.params

        def h(t, state):
            """Computes a transition rate matrix

            :param state: a tensor of shape [M, S] for S states and M population strata. States
              are S, E, I, R.  We arrange the state like this because the state vectors are then arranged
              contiguously in memory for fast calculation below.
            :return a tensor of shape [M, M, S] containing transition matric for each i=0,...,(c-1)
            """
            w_idx = tf.clip_by_value(tf.cast(t, tf.int64), 0, self.W.shape[0] - 1)
            commute_volume = tf.gather(self.W, w_idx)
            xi_idx = tf.cast(
                tf.clip_by_value(t // self.xi_freq, 0, self.params["xi"].shape[0] - 1),
                dtype=tf.int64,
            )
            xi = tf.gather(self.params["xi"], xi_idx)
            beta = self.params["beta1"] * tf.math.exp(xi)

            infec_rate = beta * (
                state[..., 2]
                + self.params["beta2"]
                * commute_volume
                * tf.linalg.matvec(self.C, state[..., 2] / self.N)
            )
            infec_rate = infec_rate / self.N + 0.000000001  # Vector of length nc

            ei = tf.broadcast_to(
                [self.params["nu"]], shape=[state.shape[0]]
            )  # Vector of length nc
            ir = tf.broadcast_to(
                [self.params["gamma"]], shape=[state.shape[0]]
            )  # Vector of length nc

            return [infec_rate, ei, ir]

        return h

    def ngm(self, t, state, param):
        """Computes a next generation matrix -- pressure from i to j is G_{ij}
        :param t: the time step
        :param state: a tensor of shape [M, S] for S states and M population strata.
                      States are S, E, I, R.
        :return: a tensor of shape [M, M] giving the expected number of new cases of
                 disease individuals in each metapopulation give rise to.
        """
        w_idx = tf.clip_by_value(tf.cast(t, tf.int64), 0, self.W.shape[0] - 1)
        commute_volume = tf.gather(self.W, w_idx)
        xi_idx = tf.cast(
            tf.clip_by_value(t // self.xi_freq, 0, self.params["xi"].shape[0] - 1),
            dtype=tf.int64,
        )
        xi = tf.gather(self.params["xi"], xi_idx)
        beta = param["beta1"] * tf.math.exp(xi)

        ngm = beta * (
            tf.eye(self.C.shape[0], dtype=state.dtype)
            + param["beta2"] * commute_volume * self.C / self.N[tf.newaxis, :]
        )
        ngm = (
            ngm
            * state[..., 0][..., tf.newaxis]
            / (self.N[:, tf.newaxis] * param["gamma"])
        )
        return ngm

    def sample(self, seed=None):
        """Runs a simulation from the epidemic model

        :param param: a dictionary of model parameters
        :param state_init: the initial state
        :returns: a tuple of times and simulated states.
        """
        hazard = self.make_h()
        t, sim = discrete_markov_simulation(
            hazard_fn=hazard,
            state=self.initial_state,
            start=self.initial_time,
            end=self.initial_time + self.num_steps * self.time_step,
            time_step=self.time_step,
            seed=seed,
        )
        return t, sim

    def log_prob(self, y):
        """Calculates the log probability of observing epidemic events y
        :param y: a list of tensors.  The first is of shape [n_times] containing times,
                  the second is of shape [n_times, n_states, n_states] containing event matrices.
        :param param: a list of parameters
        :returns: a scalar giving the log probability of the epidemic
        """
        dtype = dtype = dtype_util.common_dtype(
            [y, self.initial_state], dtype_hint=DTYPE
        )
        y = tf.convert_to_tensor(y, dtype)
        with tf.name_scope("CovidUKStochastic.log_prob"):
            hazard = self.make_h()
            return discrete_markov_log_prob(
                y, self.initial_state, hazard, self.time_step, self.stoichiometry
            )
