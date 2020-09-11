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


class CovidUKStochastic:
    def __init__(
        self,
        transition_rates,
        stoichiometry,
        initial_state,
        initial_step,
        time_delta,
        num_steps,
    ):
        """Implements a discrete-time Markov jump process for a state transition model.

        :param transition_rates: a function of the form `fn(t, state)` taking the current time `t` and state tensor `state`.  This function returns a tensor which broadcasts to the first dimension of `stoichiometry`.
        :param stoichiometry: the stochiometry matrix for the state transition model, with rows representing transitions and columns representing states.
        :param initial_state: an initial state tensor with inner dimension equal to the first dimension of `stoichiometry`.
        :param initial_step: an offset giving the time `t` of the first timestep in the model.
        :param time_delta: the size of the time step to be used.
        :param num_steps: the number of time steps across which the model runs.
        """

        self.transition_rates = transition_rates
        self.stoichiometry = stoichiometry
        self.initial_state = initial_state
        self.initial_step = initial_step
        self.time_delta = time_delta
        self.num_steps = num_steps

    def ngm(self, t, state, param):
        """Computes a next generation matrix -- pressure from i to j is G_{ij}
        :param t: the time step
        :param state: a tensor of shape [M, S] for S states and M population strata.
                      States are S, E, I, R.
        :return: a tensor of shape [M, M] giving the expected number of new cases of
                 disease individuals in each metapopulation give rise to.
        """
        t_idx = tf.clip_by_value(tf.cast(t, tf.int64), 0, self.max_t)
        commute_volume = tf.pow(tf.gather(self.W, t_idx), param["omega"])
        xi_idx = tf.gather(self.xi_select, t_idx)
        xi = tf.gather(param["xi"], xi_idx)
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
        t, sim = discrete_markov_simulation(
            hazard_fn=self.transition_rates,
            state=self.initial_state,
            start=self.initial_step,
            end=self.initial_step + self.num_steps * self.time_delta,
            time_step=self.time_delta,
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
            hazard = self.transition_rates
            return discrete_markov_log_prob(
                y, self.initial_state, hazard, self.time_delta, self.stoichiometry
            )
