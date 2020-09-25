"""Implements the COVID SEIR model as a TFP Joint Distribution"""

import tensorflow as tf
import tensorflow_probability as tfp

from covid.config import floatX
from covid.model import DiscreteTimeStateTransitionModel

tfd = tfp.distributions
DTYPE = floatX

STOICHIOMETRY = tf.constant([[-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1]])
TIME_DELTA = 1.0


def CovidUK(covariates, xi_freq, initial_state, initial_step, num_steps):
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
        sigma = tf.constant(0.01, dtype=DTYPE)
        phi = tf.constant(12.0, dtype=DTYPE)
        kernel = tfp.math.psd_kernels.MaternThreeHalves(sigma, phi)
        idx_pts = tf.cast(tf.range(num_steps // xi_freq) * xi_freq, dtype=DTYPE)
        return tfd.GaussianProcess(kernel, index_points=idx_pts[:, tf.newaxis])

    def nu():
        return tfd.Gamma(
            concentration=tf.constant(1.0, dtype=DTYPE),
            rate=tf.constant(1.0, dtype=DTYPE),
        )

    def gamma():
        return tfd.Gamma(
            concentration=tf.constant(100.0, dtype=DTYPE),
            rate=tf.constant(400.0, dtype=DTYPE),
        )

    def seir(beta1, beta2, xi, nu, gamma):

        beta1 = tf.convert_to_tensor(beta1, DTYPE)
        beta2 = tf.convert_to_tensor(beta2, DTYPE)
        xi = tf.convert_to_tensor(xi, DTYPE)
        nu = tf.convert_to_tensor(nu, DTYPE)
        gamma = tf.convert_to_tensor(gamma, DTYPE)

        def transition_rate_fn(t, state):
            C = tf.convert_to_tensor(covariates["C"], dtype=DTYPE)
            C = tf.linalg.set_diag(
                C + tf.transpose(C), tf.zeros(C.shape[0], dtype=DTYPE)
            )
            W = tf.constant(covariates["W"], dtype=DTYPE)
            N = tf.constant(covariates["pop"], dtype=DTYPE)

            w_idx = tf.clip_by_value(tf.cast(t, tf.int64), 0, W.shape[0] - 1)
            commute_volume = tf.gather(W, w_idx)
            xi_idx = tf.cast(
                tf.clip_by_value(t // 14, 0, xi.shape[0] - 1), dtype=tf.int64,
            )
            xi_ = tf.gather(xi, xi_idx)
            beta = beta1 * tf.math.exp(xi_)

            infec_rate = beta * (
                state[..., 2]
                + beta2 * commute_volume * tf.linalg.matvec(C, state[..., 2] / N)
            )
            infec_rate = infec_rate / N + 0.000000001  # Vector of length nc

            ei = tf.broadcast_to([nu], shape=[state.shape[0]])  # Vector of length nc
            ir = tf.broadcast_to([gamma], shape=[state.shape[0]])  # Vector of length nc

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
        dict(beta1=beta1, beta2=beta2, xi=xi, nu=nu, gamma=gamma, seir=seir)
    )
