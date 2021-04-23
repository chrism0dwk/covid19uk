"""Implements the COVID SEIR model as a TFP Joint Distribution"""

import pandas as pd
import numpy as np
import xarray
import tensorflow as tf
import tensorflow_probability as tfp

from gemlib.distributions import DiscreteTimeStateTransitionModel
from covid.util import impute_previous_cases
import covid.data as data

tfd = tfp.distributions

VERSION = "0.5.0"
DTYPE = np.float64

STOICHIOMETRY = np.array([[-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1]])
TIME_DELTA = 1.0
XI_FREQ = 14  # baseline transmission changes every 14 days
NU = tf.constant(0.28, dtype=DTYPE)  # E->I rate assumed known.


def gather_data(config):
    """Loads covariate data

    :param paths: a dictionary of paths to data with keys {'mobility_matrix',
                  'population_size', 'commute_volume'}
    :returns: a dictionary of covariate information to be consumed by the model
              {'C': commute_matrix, 'W': traffic_flow, 'N': population_size}
    """

    date_low = np.datetime64(config["date_range"][0])
    date_high = np.datetime64(config["date_range"][1])
    locations = data.AreaCodeData.process(config)
    mobility = data.read_mobility(
        config["mobility_matrix"], locations["lad19cd"]
    )
    popsize = data.read_population(
        config["population_size"], locations["lad19cd"]
    )
    commute_volume = data.read_traffic_flow(
        config["commute_volume"], date_low=date_low, date_high=date_high
    )

    # tier_restriction = data.TierData.process(config)[:, :, [0, 2, 3, 4]]
    dates = pd.date_range(*config["date_range"], closed="left")
    weekday = xarray.DataArray(
        dates.weekday < 5,
        name="weekday",
        dims=["time"],
        coords=[dates.to_numpy()],
    )

    cases = data.CasesData.process(config).to_xarray()
    return (
        xarray.Dataset(
            dict(
                C=mobility.astype(DTYPE),
                W=commute_volume.astype(DTYPE),
                N=popsize.astype(DTYPE),
                weekday=weekday.astype(DTYPE),
                locations=xarray.DataArray(
                    locations["name"],
                    dims=["location"],
                    coords=[locations["lad19cd"]],
                ),
            )
        ),
        xarray.Dataset(dict(cases=cases)),
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
    ei_events, lag_ei = impute_previous_cases(cases, 0.25)
    se_events, lag_se = impute_previous_cases(ei_events, 0.5)
    ir_events = np.pad(cases, ((0, 0), (lag_ei + lag_se - 2, 0)))
    ei_events = np.pad(ei_events, ((0, 0), (lag_se - 1, 0)))
    return tf.stack([se_events, ei_events, ir_events], axis=-1)


def conditional_gp(gp, observations, new_index_points):

    param = gp.parameters
    param["observation_index_points"] = param["index_points"]
    param["observations"] = observations
    param["index_points"] = new_index_points

    return tfd.GaussianProcessRegressionModel(**param)


def CovidUK(covariates, initial_state, initial_step, num_steps):
    def beta1():
        return tfd.Normal(
            loc=tf.constant(0.0, dtype=DTYPE),
            scale=tf.constant(1000.0, dtype=DTYPE),
        )

    def beta2():
        return tfd.Gamma(
            concentration=tf.constant(3.0, dtype=DTYPE),
            rate=tf.constant(10.0, dtype=DTYPE),
        )

    def sigma():
        return tfd.Gamma(
            concentration=tf.constant(2.0, dtype=DTYPE),
            rate=tf.constant(20.0, dtype=DTYPE),
        )

    def xi(beta1, sigma):
        phi = tf.constant(24.0, dtype=DTYPE)
        kernel = tfp.math.psd_kernels.MaternThreeHalves(sigma, phi)
        idx_pts = tf.cast(tf.range(num_steps // XI_FREQ) * XI_FREQ, dtype=DTYPE)
        return tfd.GaussianProcessRegressionModel(
            kernel,
            mean_fn=lambda idx: beta1,
            index_points=idx_pts[:, tf.newaxis],
        )

    def gamma0():
        return tfd.Normal(
            loc=tf.constant(0.0, dtype=DTYPE),
            scale=tf.constant(100.0, dtype=DTYPE),
        )

    def gamma1():
        return tfd.Normal(
            loc=tf.constant(0.0, dtype=DTYPE),
            scale=tf.constant(100.0, dtype=DTYPE),
        )

    def seir(beta2, xi, gamma0, gamma1):
        beta2 = tf.convert_to_tensor(beta2, DTYPE)
        xi = tf.convert_to_tensor(xi, DTYPE)
        gamma0 = tf.convert_to_tensor(gamma0, DTYPE)
        gamma1 = tf.convert_to_tensor(gamma1, DTYPE)

        C = tf.convert_to_tensor(covariates["C"], dtype=DTYPE)
        C = tf.linalg.set_diag(C, tf.zeros(C.shape[0], dtype=DTYPE))
        Cstar = C + tf.transpose(C)
        Cstar = tf.linalg.set_diag(Cstar, -tf.reduce_sum(C, axis=-2))

        W = tf.convert_to_tensor(tf.squeeze(covariates["W"]), dtype=DTYPE)
        N = tf.convert_to_tensor(tf.squeeze(covariates["N"]), dtype=DTYPE)

        weekday = tf.convert_to_tensor(covariates["weekday"], DTYPE)
        weekday = weekday - tf.reduce_mean(weekday, axis=-1)

        def transition_rate_fn(t, state):

            w_idx = tf.clip_by_value(tf.cast(t, tf.int64), 0, W.shape[0] - 1)
            commute_volume = tf.gather(W, w_idx)
            xi_idx = tf.cast(
                tf.clip_by_value(t // XI_FREQ, 0, xi.shape[0] - 1),
                dtype=tf.int64,
            )
            xi_ = tf.gather(xi, xi_idx)

            weekday_idx = tf.clip_by_value(
                tf.cast(t, tf.int64), 0, weekday.shape[0] - 1
            )
            weekday_t = tf.gather(weekday, weekday_idx)

            infec_rate = tf.math.exp(xi_) * (
                state[..., 2]
                + beta2
                * commute_volume
                * tf.linalg.matvec(Cstar, state[..., 2] / tf.squeeze(N))
            )
            infec_rate = (
                infec_rate / tf.squeeze(N) + 0.000000001
            )  # Vector of length nc

            ei = tf.broadcast_to(
                [NU], shape=[state.shape[0]]
            )  # Vector of length nc
            ir = tf.broadcast_to(
                [tf.math.exp(gamma0 + gamma1 * weekday_t)],
                shape=[state.shape[0]],
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
        dict(
            beta1=beta1,
            beta2=beta2,
            sigma=sigma,
            xi=xi,
            gamma0=gamma0,
            gamma1=gamma1,
            seir=seir,
        )
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
        C = tf.linalg.set_diag(C, tf.zeros(C.shape[0], dtype=DTYPE))
        Cstar = C + tf.transpose(C)
        Cstar = tf.linalg.set_diag(Cstar, -tf.reduce_sum(C, axis=-2))

        W = tf.constant(covar_data["W"], dtype=DTYPE)
        N = tf.constant(covar_data["N"], dtype=DTYPE)

        w_idx = tf.clip_by_value(tf.cast(t, tf.int64), 0, W.shape[0] - 1)
        commute_volume = tf.gather(W, w_idx)
        xi_idx = tf.cast(
            tf.clip_by_value(t // XI_FREQ, 0, param["xi"].shape[0] - 1),
            dtype=tf.int64,
        )
        xi = tf.gather(param["xi"], xi_idx)

        beta = tf.math.exp(xi)

        ngm = (
            beta
            * (
                tf.eye(Cstar.shape[0], dtype=state.dtype)
                + param["beta2"] * commute_volume * Cstar / N[tf.newaxis, :]
            )
            / N[:, tf.newaxis]
        )

        ngm = (1.0 - tf.math.exp(-ngm)) * state[..., 0][..., tf.newaxis]
        ngm = ngm / (1 - tf.math.exp(-tf.math.exp(param["gamma0"])))
        return ngm

    return fn
