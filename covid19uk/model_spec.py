"""Implements the COVID SEIR model as a TFP Joint Distribution"""

import pandas as pd
import geopandas as gp
import numpy as np
import xarray
import tensorflow as tf
import tensorflow_probability as tfp

from gemlib.distributions import DiscreteTimeStateTransitionModel

from covid19uk.util import impute_previous_cases

from covid19uk.data import AreaCodeData
from covid19uk.data import CasesData
from covid19uk.data import read_mobility
from covid19uk.data import read_population
from covid19uk.data import read_traffic_flow

tfd = tfp.distributions
tfd_e = tfp.experimental.distributions

DTYPE = np.float64

STOICHIOMETRY = np.array([[-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1]])
TIME_DELTA = 1.0
NU = tf.constant(0.28, dtype=DTYPE)  # E->I rate assumed known.


def _compute_adjacency_matrix(geom, names, tol=200):
    mat = geom.apply(lambda x: geom.distance(x) < tol).to_numpy()
    np.fill_diagonal(mat, False)

    # Fix for islands > tol apart
    num_neighbours = mat.sum(axis=-1)
    islands = np.where(num_neighbours == 0)[0]
    closest_neighbour = [
        geom.distance(geom.iloc[i]).argsort()[1] for i in islands
    ]
    mat[islands, closest_neighbour] = True
    mat = mat | mat.T  # Ensure symmetry

    return xarray.DataArray(
        mat.astype(DTYPE),  # Coerce to global float type
        coords=[names, names],
        dims=["location_dest", "location_src"],
    )


def gather_data(config):
    """Loads covariate data

    :param paths: a dictionary of paths to data with keys {'mobility_matrix',
                  'population_size', 'commute_volume'}
    :returns: a dictionary of covariate information to be consumed by the model
              {'C': commute_matrix, 'W': traffic_flow, 'N': population_size}
    """

    date_low = np.datetime64(config["date_range"][0])
    date_high = np.datetime64(config["date_range"][1])
    locations = AreaCodeData.process(config)
    mobility = read_mobility(config["mobility_matrix"], locations["lad19cd"])
    popsize = read_population(config["population_size"], locations["lad19cd"])
    commute_volume = read_traffic_flow(
        config["commute_volume"], date_low=date_low, date_high=date_high
    )
    geo = gp.read_file(config["geopackage"], layer="UK2019mod_pop_xgen")
    geo = geo.sort_values("lad19cd")
    geo = geo[geo["lad19cd"].isin(locations["lad19cd"])]

    adjacency = _compute_adjacency_matrix(geo.geometry, geo["lad19cd"], 200)

    area = xarray.DataArray(
        geo.area,
        name="area",
        dims=["location"],
        coords=[geo["lad19cd"]],
    )

    dates = pd.date_range(*config["date_range"], closed="left")
    weekday = xarray.DataArray(
        dates.weekday < 5,
        name="weekday",
        dims=["time"],
        coords=[dates.to_numpy()],
    )

    cases = CasesData.process(config).to_xarray()
    return (
        xarray.Dataset(
            dict(
                C=mobility.astype(DTYPE),
                W=commute_volume.astype(DTYPE),
                N=popsize.astype(DTYPE),
                adjacency=adjacency,
                weekday=weekday.astype(DTYPE),
                area=area.astype(DTYPE),
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
    def alpha_0():
        return tfd.Normal(
            loc=tf.constant(0.0, dtype=DTYPE),
            scale=tf.constant(10.0, dtype=DTYPE),
        )

    def beta_area():
        return tfd.Normal(
            loc=tf.constant(0.0, dtype=DTYPE),
            scale=tf.constant(1.0, dtype=DTYPE),
        )

    def psi():
        return tfd.Gamma(
            concentration=tf.constant(3.0, dtype=DTYPE),
            rate=tf.constant(10.0, dtype=DTYPE),
        )

    def alpha_t():
        """Time-varying force of infection"""
        return tfd.MultivariateNormalDiag(
            loc=tf.constant(0.0, dtype=DTYPE),
            scale_diag=tf.fill(
                [num_steps - 1], tf.constant(0.005, dtype=DTYPE)
            ),
        )

    def sigma_space():
        """Variance of CAR prior on space"""
        # return tfd.HalfNormal(scale=tf.constant(0.1, dtype=DTYPE))
        return tfd.InverseGamma(
            concentration=tf.constant(2.0, dtype=DTYPE),
            scale=tf.constant(0.5, dtype=DTYPE),
        )

    def rho():
        """Correlation between neighbouring regions"""
        return tfd.Beta(
            concentration0=tf.constant(2.0, dtype=DTYPE),
            concentration1=tf.constant(2.0, dtype=DTYPE),
        )

    def spatial_effect(rho):
        W = tf.convert_to_tensor(covariates["adjacency"])
        Dw = tf.linalg.diag(tf.reduce_sum(W, axis=-1))  # row sums
        precision = Dw - rho * W
        precision_factor = tf.linalg.cholesky(precision)
        return tfd_e.MultivariateNormalPrecisionFactorLinearOperator(
            loc=tf.constant(0.0, DTYPE),
            precision_factor=tf.linalg.LinearOperatorFullMatrix(
                precision_factor
            ),
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

    def seir(
        psi,
        beta_area,
        alpha_0,
        alpha_t,
        spatial_effect,
        sigma_space,
        gamma0,
        gamma1,
    ):
        psi = tf.convert_to_tensor(psi, DTYPE)
        beta_area = tf.convert_to_tensor(beta_area, DTYPE)
        alpha_t = tf.convert_to_tensor(alpha_t, DTYPE)
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

        # Area in 100km^2
        area = tf.convert_to_tensor(covariates["area"], DTYPE)
        log_area = tf.math.log(area / 100000000.0)  # log area in 100km^2
        log_area = log_area - tf.reduce_mean(log_area)

        def transition_rate_fn(t, state):

            w_idx = tf.clip_by_value(tf.cast(t, tf.int64), 0, W.shape[0] - 1)
            commute_volume = tf.gather(W, w_idx)

            weekday_idx = tf.clip_by_value(
                tf.cast(t, tf.int64), 0, weekday.shape[0] - 1
            )
            weekday_t = tf.gather(weekday, weekday_idx)

            with tf.name_scope("Pick_alpha_t"):
                b_t = alpha_0 + tf.cumsum(alpha_t)
                alpha_t_idx = tf.cast(t, tf.int64)
                alpha_t_ = tf.where(
                    alpha_t_idx == 0,
                    alpha_0,
                    tf.gather(
                        b_t,
                        tf.clip_by_value(
                            alpha_t_idx - 1,
                            clip_value_min=0,
                            clip_value_max=alpha_t.shape[0] - 1,
                        ),
                    ),
                )
            eta = alpha_t_ + beta_area * log_area + sigma_space * spatial_effect
            infec_rate = tf.math.exp(eta) * (
                state[..., 2]
                + psi
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
            alpha_0=alpha_0,
            beta_area=beta_area,
            psi=psi,
            alpha_t=alpha_t,
            sigma_space=sigma_space,
            rho=rho,
            spatial_effect=spatial_effect,
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

        # Area in 100km^2
        area = tf.convert_to_tensor(covar_data["area"], DTYPE)
        log_area = tf.math.log(area / 100000000.0)  # log area in 100km^2
        log_area = log_area - tf.reduce_mean(log_area)

        w_idx = tf.clip_by_value(tf.cast(t, tf.int64), 0, W.shape[0] - 1)
        commute_volume = tf.gather(W, w_idx)
        b_t = param["alpha_0"] + tf.cumsum(param["alpha_t"])
        alpha_t_ = tf.where(
            t == 0,
            param["alpha_0"],
            tf.gather(
                b_t,
                tf.clip_by_value(
                    t,
                    clip_value_min=0,
                    clip_value_max=param["alpha_t"].shape[-1] - 1,
                ),
            ),
        )

        eta = (
            alpha_t_
            + param["beta_area"] * log_area[:, tf.newaxis]
            + param["sigma_space"] * param["spatial_effect"]
        )
        infec_rate = (
            tf.math.exp(eta)
            * (
                tf.eye(Cstar.shape[0], dtype=state.dtype)
                + param["psi"] * commute_volume * Cstar / N[tf.newaxis, :]
            )
            / N[:, tf.newaxis]
        )
        infec_prob = 1.0 - tf.math.exp(-infec_rate)

        expected_new_infec = infec_prob * state[..., 0][..., tf.newaxis]
        expected_infec_period = 1.0 / (
            1.0 - tf.math.exp(-tf.math.exp(param["gamma0"]))
        )
        ngm = expected_new_infec * expected_infec_period
        return ngm

    return fn
