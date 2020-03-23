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

DTYPE = np.float64


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
    scale_tril = tf.linalg.cholesky(covariance)
    rv = tfp.distributions.MultivariateNormalTriL(loc=tf.zeros(covariance.shape[0], dtype=tf.float64),
                                                  scale_tril=scale_tril)

    def _fn(state_parts, seed):
        with tf.name_scope(name or 'random_walk_mvnorm_fn'):
            new_state_parts = [rv.sample() + state_part for state_part in state_parts]
            return new_state_parts

    return _fn



if __name__ == '__main__':

    parser = optparse.OptionParser()
    parser.add_option("--config", "-c", dest="config", default="ode_config.yaml",
                      help="configuration file")
    options, args = parser.parse_args()
    with open(options.config, 'r') as ymlfile:
        config = yaml.load(ymlfile)

    K_tt, age_groups = load_age_mixing(config['data']['age_mixing_matrix_term'])
    K_hh, _ = load_age_mixing(config['data']['age_mixing_matrix_hol'])

    T, la_names = load_mobility_matrix(config['data']['mobility_matrix'])
    np.fill_diagonal(T, 0.)

    N, n_names = load_population(config['data']['population_size'])

    K_tt = K_tt.astype(DTYPE)
    K_hh = K_hh.astype(DTYPE)
    T = T.astype(DTYPE)
    N = N.astype(DTYPE)

    param = sanitise_parameter(config['parameter'])
    settings = sanitise_settings(config['settings'])

    case_reports = pd.read_csv(config['data']['reported_cases'])
    case_reports['DateVal'] = pd.to_datetime(case_reports['DateVal'])
    case_reports = case_reports[case_reports['DateVal'] >= '2020-02-19']
    date_range = [case_reports['DateVal'].min(), case_reports['DateVal'].max()]
    y = case_reports['CumCases'].to_numpy()
    y_incr = np.round((y[1:] - y[:-1]) * 0.8)

    simulator = CovidUKODE(
        M_tt=K_tt,
        M_hh=K_hh,
        C=T,
        N=N,
        start=date_range[0]-np.timedelta64(1,'D'),
        end=date_range[1],
        holidays=settings['holiday'],
        bg_max_t=settings['bg_max_time'],
        t_step=int(settings['time_step']))

    seeding = seed_areas(N, n_names)  # Seed 40-44 age group, 30 seeds by popn size
    state_init = simulator.create_initial_state(init_matrix=seeding)

    def logp(par):
        p = param
        p['beta1'] = par[0]
        p['gamma'] = par[1]
        beta_logp = tfd.Gamma(concentration=tf.constant(1., tf.float64), rate=tf.constant(1., tf.float64)).log_prob(p['beta1'])
        gamma_logp = tfd.Gamma(concentration=tf.constant(100., tf.float64), rate=tf.constant(400., tf.float64)).log_prob(p['gamma'])
        t, sim, solve = simulator.simulate(p, state_init)
        y_logp = covid19uk_logp(y_incr, sim, 0.1)
        logp = beta_logp + gamma_logp + tf.reduce_sum(y_logp)
        return logp

    def trace_fn(_, pkr):
      return (
          pkr.inner_results.log_accept_ratio,
          pkr.inner_results.accepted_results.target_log_prob,
          pkr.inner_results.accepted_results.step_size)


    unconstraining_bijector = [tfb.Exp()]
    initial_mcmc_state = np.array([0.05, 0.25], dtype=np.float64)
    print("Initial log likelihood:", logp(initial_mcmc_state))

    @tf.function(autograph=False, experimental_compile=True)
    def sample(n_samples, init_state, scale, num_burnin_steps=0):
        return tfp.mcmc.sample_chain(
            num_results=n_samples,
            num_burnin_steps=num_burnin_steps,
            current_state=init_state,
            kernel=tfp.mcmc.TransformedTransitionKernel(
                    inner_kernel=tfp.mcmc.RandomWalkMetropolis(
                        target_log_prob_fn=logp,
                        new_state_fn=random_walk_mvnorm_fn(scale)
                    ),
                    bijector=unconstraining_bijector),
            trace_fn=lambda _, pkr: pkr.inner_results.is_accepted)

    joint_posterior = tf.zeros([0] + list(initial_mcmc_state.shape), dtype=DTYPE)

    scale = np.diag([0.1, 0.1])
    overall_start = time.perf_counter()

    num_covariance_estimation_iterations = 50
    num_covariance_estimation_samples = 50
    num_final_samples = 10000
    with tf.device("/CPU:0"):
        start = time.perf_counter()
        for i in range(num_covariance_estimation_iterations):
            step_start = time.perf_counter()
            samples, results = sample(num_covariance_estimation_samples,
                                      initial_mcmc_state,
                                      scale)
            step_end = time.perf_counter()
            print(f'{i} time {step_end - step_start}')
            print("Acceptance: ", results.numpy().mean())
            joint_posterior = tf.concat([joint_posterior, samples], axis=0)
            cov = tfp.stats.covariance(tf.math.log(joint_posterior))
            print(cov.numpy())
            scale = cov * 2.38**2 / joint_posterior.shape[1]
            initial_mcmc_state = joint_posterior[-1, :]

        step_start = time.perf_counter()
        samples, results = sample(num_final_samples,
                                  init_state=joint_posterior[-1, :], scale=scale,)
        joint_posterior = tf.concat([joint_posterior, samples], axis=0)
        step_end = time.perf_counter()
        print(f'Sampling step time {step_end - step_start}')
        end = time.perf_counter()
        print(f"Simulation complete in {end-start} seconds")
        print("Acceptance: ", np.mean(results.numpy()))
        print(tfp.stats.covariance(tf.math.log(joint_posterior)))

    fig, ax = plt.subplots(1, 3)
    ax[0].plot(joint_posterior[:, 0])
    ax[1].plot(joint_posterior[:, 1])
    plt.show()
    print(f"Posterior mean: {np.mean(joint_posterior, axis=0)}")

    with open('pi_beta_2020-03-15.pkl', 'wb') as f:
        pkl.dump(joint_posterior, f)
