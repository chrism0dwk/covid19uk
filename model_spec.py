"""Implements the COVID SEIR model as a TFP Joint Distribution"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from gemlib.distributions import DiscreteTimeStateTransitionModel
from covid.util import impute_previous_cases
import covid.data as data

tfd = tfp.distributions
DTYPE = np.float64

STOICHIOMETRY = tf.constant([[-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1]])
TIME_DELTA = 1.0
XI_FREQ = 14  # baseline transmission changes every 14 days
NU = tf.constant(0.5, dtype=DTYPE)  # E->I rate assumed known.


def read_covariates(paths, date_low, date_high):
    """Loads covariate data

    :param paths: a dictionary of paths to data with keys {'mobility_matrix',
                  'population_size', 'commute_volume'}
    :returns: a dictionary of covariate information to be consumed by the model
              {'C': commute_matrix, 'W': traffic_flow, 'N': population_size}
    """
    mobility = data.read_mobility(paths["mobility_matrix"])
    popsize = data.read_population(paths["population_size"])
    commute_volume = data.read_traffic_flow(
        paths["commute_volume"], date_low=date_low, date_high=date_high
    )

    return dict(
        C=mobility.to_numpy().astype(DTYPE),
        W=commute_volume.to_numpy().astype(DTYPE),
        N=popsize.to_numpy().astype(DTYPE),
    )


def impute_censored_events(cases):
    """Imputes censored S->E and E->I events using geometric
       sampling algorithm in `impute_previous_cases`

    There are application-specific magic numbers hard-coded below,
    which reflect experimentation to get the right lag between EI and
    IR events, and SE and EI events respectively.  These were chosen
    by experimentation and examination of the resulting epidemic
    trajectories.

    :param cases: a MxT matrix of case numbers (I->R)
    :returns: a MxTx3 tensor of events where the first two indices of
              the right-most dimension contain the imputed event times.
    """
    ei_events, lag_ei = impute_previous_cases(cases, 0.44)
    se_events, lag_se = impute_previous_cases(ei_events, 2.0)
    ir_events = np.pad(cases, ((0, 0), (lag_ei + lag_se - 2, 0)))
    ei_events = np.pad(ei_events, ((0, 0), (lag_se - 1, 0)))
    return tf.stack([se_events, ei_events, ir_events], axis=-1)


def CovidUK(covariates, initial_state, initial_step, num_steps):
    def beta1():
        return tfd.Gamma(
            concentration=tf.constant(1.0, dtype=DTYPE),
            rate=tf.constant(1.0, dtype=DTYPE),
        )

    def beta2():
        return tfd.Gamma(
            concentration=tf.constant(3.0, dtype=DTYPE),
            rate=tf.constant(10.0, dtype=DTYPE),
        )

    def xi():
        sigma = tf.constant(0.4, dtype=DTYPE)
        phi = tf.constant(24.0, dtype=DTYPE)
        kernel = tfp.math.psd_kernels.MaternThreeHalves(sigma, phi)
        idx_pts = tf.cast(tf.range(num_steps // XI_FREQ) * XI_FREQ, dtype=DTYPE)
        return tfd.GaussianProcess(kernel, index_points=idx_pts[:, tf.newaxis])

    def gamma():
        return tfd.Gamma(
            concentration=tf.constant(100.0, dtype=DTYPE),
            rate=tf.constant(400.0, dtype=DTYPE),
        )

    def seir(beta1, beta2, xi, gamma):

        beta1 = tf.convert_to_tensor(beta1, DTYPE)
        beta2 = tf.convert_to_tensor(beta2, DTYPE)
        xi = tf.convert_to_tensor(xi, DTYPE)
        gamma = tf.convert_to_tensor(gamma, DTYPE)

        def transition_rate_fn(t, state):
            C = tf.convert_to_tensor(covariates["C"], dtype=DTYPE)
            C = tf.linalg.set_diag(
                C + tf.transpose(C), tf.zeros(C.shape[0], dtype=DTYPE)
            )
            W = tf.constant(np.squeeze(covariates["W"]), dtype=DTYPE)
            N = tf.constant(np.squeeze(covariates["N"]), dtype=DTYPE)

            w_idx = tf.clip_by_value(tf.cast(t, tf.int64), 0, W.shape[0] - 1)
            commute_volume = tf.gather(W, w_idx)
            xi_idx = tf.cast(
                tf.clip_by_value(t // XI_FREQ, 0, xi.shape[0] - 1),
                dtype=tf.int64,
            )
            xi_ = tf.gather(xi, xi_idx)
            beta = beta1 * tf.math.exp(xi_)

            infec_rate = beta * (
                state[..., 2]
                + beta2
                * commute_volume
                * tf.linalg.matvec(C, state[..., 2] / tf.squeeze(N))
            )
            infec_rate = (
                infec_rate / tf.squeeze(N) + 0.000000001
            )  # Vector of length nc

            ei = tf.broadcast_to(
                [NU], shape=[state.shape[0]]
            )  # Vector of length nc
            ir = tf.broadcast_to(
                [gamma], shape=[state.shape[0]]
            )  # Vector of length nc

            return [infec_rate, ei, ir]

        return DiscreteTimeStateTransitionModel(
            transition_rates=transition_rate_fn,
            stoichiometry=STOICHIOMETRY,
            initial_state=initial_state,
            initial_step=initial_step,
            time_delta=TIME_DELTA,
            num_steps=num_steps,
        )

    return tfd.JointDistributionNamed(
        dict(beta1=beta1, beta2=beta2, xi=xi, gamma=gamma, seir=seir)
    )


def next_generation_matrix_fn(covar_data, param):
    """The next generation matrix calculates the force of infection from
    individuals in metapopulation i to all other metapopulations j during
    a typical infectious period (1/gamma). i.e.

      \[ A_{ij} = S_j * \beta_1 ( 1 + \beta_2 * w_t * C_{ij} / N_i) / N_j / gamma \]

    :param covar_data: a dictionary of covariate data
    :param param: a dictionary of parameters
    :returns: a function taking arguments `t` and `state` giving the time and
              epidemic state (SEIR) for which the NGM is to be calculated.  This
              function in turn returns an MxM next generation matrix.
    """

    def fn(t, state):
        C = tf.convert_to_tensor(covar_data["C"], dtype=DTYPE)
        C = tf.linalg.set_diag(
            C + tf.transpose(C), tf.zeros(C.shape[0], dtype=DTYPE)
        )
        W = tf.constant(covar_data["W"], dtype=DTYPE)
        N = tf.constant(covar_data["N"], dtype=DTYPE)

        w_idx = tf.clip_by_value(tf.cast(t, tf.int64), 0, W.shape[0] - 1)
        commute_volume = tf.gather(W, w_idx)
        xi_idx = tf.cast(
            tf.clip_by_value(t // XI_FREQ, 0, param["xi"].shape[0] - 1),
            dtype=tf.int64,
        )
        xi = tf.gather(param["xi"], xi_idx)
        beta = param["beta1"] * tf.math.exp(xi)

        ngm = beta * (
            tf.eye(C.shape[0], dtype=state.dtype)
            + param["beta2"] * commute_volume * C / N[tf.newaxis, :]
        )
        ngm = (
            ngm
            * state[..., 0][..., tf.newaxis]
            / (N[:, tf.newaxis] * param["gamma"])
        )
        return ngm

    return fn
