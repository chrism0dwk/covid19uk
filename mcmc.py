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
    def logp(epsilon, beta):
        p = param
        p['epsilon'] = epsilon
        p['beta1'] = beta
        epsilon_logp = tfd.Gamma(concentration=1., rate=1.).log_prob(p['epsilon'])
        beta_logp = tfd.Gamma(concentration=1., rate=1.).log_prob(p['beta1'])
        t, sim, solve = simulator.simulate(p, state_init)
        y_logp = covid19uk_logp(y_incr, sim, 0.1)
        logp = epsilon_logp + beta_logp + tf.reduce_sum(y_logp)
        return logp

    def trace_fn(_, pkr):
      return (
          pkr.inner_results.log_accept_ratio,
          pkr.inner_results.accepted_results.target_log_prob,
          pkr.inner_results.accepted_results.step_size)


    unconstraining_bijector = [tfb.Exp(), tfb.Exp()]
    initial_mcmc_state = [0.00001, 0.03]
    print("Initial log likelihood:", logp(*initial_mcmc_state))

    @tf.function
    def sample(n_samples, step_size=[0.01, 0.1]):
        return tfp.mcmc.sample_chain(
            num_results=n_samples,
            num_burnin_steps=0,
            current_state=initial_mcmc_state,
            kernel=tfp.mcmc.SimpleStepSizeAdaptation(
                tfp.mcmc.TransformedTransitionKernel(
                    inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
                        target_log_prob_fn=logp,
                        step_size=step_size,
                        num_leapfrog_steps=5),
                    bijector=unconstraining_bijector),
                num_adaptation_steps=500),
            trace_fn=lambda _, pkr: pkr.inner_results.inner_results.accepted_results.step_size)

    with tf.device("/CPU:0"):
        start = time.perf_counter()
        pi_beta, results = sample(1000, [0.1, 0.1])
        # sd = tfp.stats.stddev(pi_beta, sample_axis=1)
        # sd = sd / tf.reduce_sum(sd)
        # new_step_size = [results[-1] * sd[0], results[-1] * sd[1]]
        # print("New step size:",new_step_size)
        # pi_beta, results = sample(1000, step_size=new_step_size)
        end = time.perf_counter()
        print(f"Simulation complete in {end-start} seconds")
        print(results)



    fig, ax = plt.subplots(1, 2)
    ax[0].plot(pi_beta[0])
    ax[1].plot(pi_beta[1])
    plt.show()
    print(f"Posterior mean: {np.mean(pi_beta[0])}")

    with open('pi_beta_2020-03-15.pkl', 'wb') as f:
        pkl.dump(pi_beta, f)

    #dates = settings['start'] + t.numpy().astype(np.timedelta64)
    #plotting(dates, sim)
