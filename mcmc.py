import optparse
import yaml
import time
import pickle as pkl
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
import matplotlib.pyplot as plt
from tensorflow_probability.python.util import SeedStream

from covid.rdata import load_population, load_age_mixing, load_mobility_matrix
from covid.model import CovidUKODE, covid19uk_logp
from covid.util import *


def plotting(dates, sim):
    print("Initial R0:", simulator.eval_R0(param))
    print("Doubling time:", doubling_time(dates, sim.numpy(), '2020-02-27','2020-03-13'))

    fig = plt.figure()
    removals = tf.reduce_sum(sim[:, 3, :], axis=1)
    infected = tf.reduce_sum(sim[:, 1:3, :], axis=[1,2])
    exposed = tf.reduce_sum(sim[:, 1, :], axis=1)
    date = np.squeeze(np.where(dates == np.datetime64('2020-03-13'))[0])
    print("Daily incidence 2020-03-13:", exposed[date]-exposed[date-1])

    plt.plot(dates, removals*0.10, '-', label='10% reporting')
    plt.plot(dates, infected, '-', color='red', label='Total infected')
    plt.plot(dates, removals, '-', color='gray', label='Total recovered/detected/died')

    plt.scatter(np.datetime64('2020-03-13'), 600, label='gov.uk cases 13th March 2020')
    plt.legend()
    plt.grid(True)
    fig.autofmt_xdate()
    plt.show()


def random_walk_mvnorm_fn(covariance, name=None):
    """Returns callable that adds Multivariate Normal noise to the input"""
    covariance = covariance + tf.eye(covariance.shape[0], dtype=tf.float64) * 1.e-9

    rv = tfp.distributions.MultivariateNormalTriL(loc=tf.zeros(covariance.shape[0], dtype=tf.float64),
                                                  scale_tril=tf.linalg.cholesky(
                                                      tf.convert_to_tensor(covariance,
                                                                           dtype=tf.float64)))

    def _fn(state_parts, seed):
        with tf.name_scope(name or 'random_walk_mvnorm_fn'):
            new_state_parts = rv.sample() + state_parts
            return new_state_parts

    return _fn



if __name__ == '__main__':

    parser = optparse.OptionParser()
    parser.add_option("--config", "-c", dest="config",
                      help="configuration file")
    options, args = parser.parse_args()
    with open(options.config, 'r') as ymlfile:
        config = yaml.load(ymlfile)

    K_tt, age_groups = load_age_mixing(config['data']['age_mixing_matrix_term'])
    K_hh, _ = load_age_mixing(config['data']['age_mixing_matrix_hol'])

    T, la_names = load_mobility_matrix(config['data']['mobility_matrix'])
    np.fill_diagonal(T, 0.)

    N, n_names = load_population(config['data']['population_size'])

    param = sanitise_parameter(config['parameter'])
    settings = sanitise_settings(config['settings'])

    case_reports = pd.read_csv(config['data']['reported_cases'])
    case_reports['DateVal'] = pd.to_datetime(case_reports['DateVal'])
    date_range = [case_reports['DateVal'].min(), case_reports['DateVal'].max()]
    y = case_reports['CumCases'].to_numpy()
    y_incr = np.round((y[1:] - y[:-1]) * 0.8)

    simulator = CovidUKODE(K_tt, K_hh, T, N, date_range[0]-np.timedelta64(1,'D'),
                           date_range[1], settings['holiday'], settings['bg_max_time'], int(settings['time_step']))

    seeding = seed_areas(N, n_names)  # Seed 40-44 age group, 30 seeds by popn size
    state_init = simulator.create_initial_state(init_matrix=seeding)

    #@tf.function
    def logp(par):
        p = param
        p['epsilon'] = par[0]
        p['beta1'] = par[1]
        p['gamma'] = par[2]
        epsilon_logp = tfd.Gamma(concentration=tf.constant(1., tf.float64), rate=tf.constant(1., tf.float64)).log_prob(p['epsilon'])
        beta_logp = tfd.Gamma(concentration=tf.constant(1., tf.float64), rate=tf.constant(1., tf.float64)).log_prob(p['beta1'])
        gamma_logp = tfd.Gamma(concentration=tf.constant(100., tf.float64), rate=tf.constant(400., tf.float64)).log_prob(p['gamma'])
        t, sim, solve = simulator.simulate(p, state_init)
        y_logp = covid19uk_logp(y_incr, sim, 0.1)
        logp = epsilon_logp + beta_logp + gamma_logp + tf.reduce_sum(y_logp)
        return logp

    def trace_fn(_, pkr):
      return (
          pkr.inner_results.log_accept_ratio,
          pkr.inner_results.accepted_results.target_log_prob,
          pkr.inner_results.accepted_results.step_size)


    unconstraining_bijector = [tfb.Exp()]
    initial_mcmc_state = np.array([0.001,  0.036, 0.25], dtype=np.float64)
    print("Initial log likelihood:", logp(initial_mcmc_state))

    @tf.function(experimental_compile=True)
    def sample(n_samples, init_state, scale):
        return tfp.mcmc.sample_chain(
            num_results=n_samples,
            num_burnin_steps=0,
            current_state=init_state,
            kernel=tfp.mcmc.TransformedTransitionKernel(
                    inner_kernel=tfp.mcmc.RandomWalkMetropolis(
                        target_log_prob_fn=logp,
                        new_state_fn=random_walk_mvnorm_fn(scale)
                    ),
                    bijector=unconstraining_bijector),
            trace_fn=lambda _, pkr: pkr.inner_results.is_accepted)

    with tf.device("/CPU:0"):
        cov = np.diag([0.00001, 0.00001, 0.00001])
        start = time.perf_counter()
        joint_posterior, results = sample(50, init_state=initial_mcmc_state, scale=cov)
        for i in range(200):
            cov = tfp.stats.covariance(tf.math.log(joint_posterior)) * 2.38**2 / joint_posterior.shape[1]
            print(cov.numpy())
            posterior_new, results = sample(50, joint_posterior[-1, :], cov)
            joint_posterior = tf.concat([joint_posterior, posterior_new], axis=0)
        #posterior_new, results = sample(2000, init_state=joint_posterior[-1, :], scale=cov)
        #joint_posterior = tf.concat([joint_posterior, posterior_new], axis=0)
        end = time.perf_counter()
        print(f"Simulation complete in {end-start} seconds")
        print("Acceptance: ", np.mean(results.numpy()))
        print(tfp.stats.covariance(tf.math.log(joint_posterior)))

    fig, ax = plt.subplots(1, 3)
    ax[0].plot(joint_posterior[:, 0])
    ax[1].plot(joint_posterior[:, 1])
    ax[2].plot(joint_posterior[:, 2])
    plt.show()
    print(f"Posterior mean: {np.mean(joint_posterior, axis=0)}")

    with open('pi_beta_2020-03-15.pkl', 'wb') as f:
        pkl.dump(joint_posterior, f)

    #dates = settings['start'] + t.numpy().astype(np.timedelta64)
    #plotting(dates, sim)
