import unittest
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from covid.impl.discrete_markov import discrete_markov_simulation, discrete_markov_log_prob
from covid.impl.util import make_transition_matrix


class TestChainBinomialLogp(unittest.TestCase):

    def setUp(self) -> None:
        self.init_state = tf.constant([[9, 1, 0],
                                       [5, 1, 0],
                                       [3, 0, 0],
                                       [6, 2, 0]], dtype=tf.float32)
        seed_stream = tfp.util.SeedStream(1., salt='prng')
        self.epidemic = discrete_markov_simulation(self.hazard_fn(0.4, 0.14), self.init_state, 0., 25., 1., seed=seed_stream)

    def hazard_fn(self, beta, gamma) -> tf.Tensor:
        """Hazard rates for state transitions.
        :param state: a [m, nc, ns] tensor for m batches of nc metapopulations in ns states
        :return a tensor of rates
        """
        def fn(t: tf.Tensor, state: tf.Tensor) -> tf.Tensor:
            S, I, R = range(state.shape[-1])

            si = beta * state[..., S] * state[..., I] / tf.reduce_sum(state, axis=-1)
            ir = gamma * state[..., I]

            rate_matrix = make_transition_matrix([si, ir], [[S, I], [I, R]], state.shape)
            return rate_matrix
        return fn

    def test_chainbinomial_logp(self) -> None:

        hazard = self.hazard_fn(0.4, 0.14)
        logp = discrete_markov_log_prob(self.epidemic[1][:2], self.init_state, hazard, 1.0)
        print(logp)

    def test_discrete_markov_mcmc(self) -> None:

        def logp_fn(beta, gamma):
            logp_beta = tfd.Gamma(0.1, 0.1).log_prob(beta)
            logp_gamma = tfd.Gamma(0.1, 0.1).log_prob(gamma)
            hazard = self.hazard_fn(beta, gamma)
            logp_epi = discrete_markov_log_prob(self.epidemic[1], self.init_state, hazard, 1.0)
            return logp_beta + logp_gamma + logp_epi

        unconstraining_bijector = [tfb.Exp(), tfb.Exp()]

        @tf.function  #(autograph=False, experimental_compile=True)
        def sample(n_samples, init_state, scale, num_burnin_steps=0):
            return tfp.mcmc.sample_chain(
                num_results=n_samples,
                num_burnin_steps=num_burnin_steps,
                current_state=init_state,
                kernel=tfp.mcmc.TransformedTransitionKernel(
                    inner_kernel=tfp.mcmc.RandomWalkMetropolis(
                        target_log_prob_fn=logp_fn,
                        new_state_fn=tfp.mcmc.random_walk_normal_fn(scale=scale)
                    ),
                    bijector=unconstraining_bijector),
                trace_fn=lambda _, pkr: pkr.inner_results.is_accepted)

        samples, results = sample(5000, init_state=[tf.constant(0.2), tf.constant(0.2)],
                                  scale=[tf.constant(0.4), tf.constant(0.4)])
        accept = np.mean(results.numpy(), axis=0)
        print("Accept:", accept)
        pihat = tf.reduce_mean(samples, axis=1)
        q = tfp.stats.percentile(samples, q=[2.5, 97.5], axis=1)
        print("piHat:", pihat.numpy())
        print("Percentiles:", q.numpy())


if __name__ == '__main__':
    unittest.main()
