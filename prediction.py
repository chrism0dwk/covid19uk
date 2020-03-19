"""Prediction functions"""
import optparse
import yaml
import pickle as pkl
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow_probability import stats as tfs
import matplotlib.pyplot as plt

from covid.model import CovidUKODE
from covid.rdata import load_age_mixing, load_mobility_matrix, load_population
from covid.util import sanitise_settings, sanitise_parameter, seed_areas


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
    param['epsilon'] = 0.0
    settings = sanitise_settings(config['settings'])

    case_reports = pd.read_csv(config['data']['reported_cases'])
    case_reports['DateVal'] = pd.to_datetime(case_reports['DateVal'])
    date_range = [case_reports['DateVal'].min(), case_reports['DateVal'].max()]
    y = case_reports['CumCases'].to_numpy()
    y_incr = y[1:] - y[-1:]

    with open('pi_beta_2020-03-15.pkl', 'rb') as f:
        pi_beta = pkl.load(f)

    # Predictive distribution of epidemic spread
    data_dates = np.arange(date_range[0],
                           date_range[1]+np.timedelta64(1,'D'),
                           np.timedelta64(1, 'D'))
    simulator = CovidUKODE(K_tt, K_hh, T, N, date_range[0] - np.timedelta64(1, 'D'),
                           np.datetime64('2020-05-01'), settings['holiday'], settings['bg_max_time'], 1)
    seeding = seed_areas(N, n_names)  # Seed 40-44 age group, 30 seeds by popn size
    state_init = simulator.create_initial_state(init_matrix=seeding)

    @tf.function
    def prediction(beta):
        sims = tf.TensorArray(tf.float32, size=beta.shape[0])
        for i in tf.range(beta.shape[0]):
            p = param
            p['beta1'] = beta[i]
            t, sim, solver_results = simulator.simulate(p, state_init)
            sim_aggr = tf.reduce_sum(sim, axis=2)
            sims = sims.write(i, sim_aggr)
        return sims.gather(range(beta.shape[0]))

    draws = pi_beta[0].numpy()[np.arange(0, pi_beta[0].shape[0], 20)]
    sims = prediction(draws)

    dates = np.arange(date_range[0]-np.timedelta64(1, 'D'), np.datetime64('2020-05-01'),
                      np.timedelta64(1, 'D'))
    total_infected = tfs.percentile(tf.reduce_sum(sims[:, :, 1:3], axis=2), q=[2.5, 50, 97.5], axis=0)
    removed = tfs.percentile(sims[:, :, 3], q=[2.5, 50, 97.5], axis=0)
    removed_observed = tfs.percentile(removed * 0.1, q=[2.5, 50, 97.5], axis=0)

    fig = plt.figure()
    filler = plt.fill_between(dates, total_infected[0, :], total_infected[2, :], color='lightgray', label="95% credible interval")
    plt.fill_between(dates, removed[0, :], removed[2, :], color='lightgray')
    plt.fill_between(dates, removed_observed[0, :], removed_observed[2, :], color='lightgray')
    ti_line = plt.plot(dates, total_infected[1, :], '-', color='red', alpha=0.4, label="Infected")
    rem_line = plt.plot(dates, removed[1, :], '-', color='blue', label="Removed")
    ro_line = plt.plot(dates, removed_observed[1, :], '-', color='orange', label='Predicted detections')
    marks = plt.plot(data_dates, y, '+', label='Observed cases')
    plt.legend([ti_line[0], rem_line[0], ro_line[0], filler, marks[0]],
               ["Infected", "Removed", "Predicted detections", "95% credible interval", "Observed counts"])
    plt.grid()
    plt.xlabel("Date")
    plt.ylabel("$10^7$ individuals")
    fig.autofmt_xdate()
    plt.show()

    # Number of new cases per day
    new_cases = tfs.percentile(removed[:, 1:] - removed[:, :-1],  q=[2.5, 50, 97.5], axis=0)/10000.
    fig = plt.figure()
    plt.fill_between(dates[:-1], new_cases[0, :], new_cases[2, :], color='lightgray', label="95% credible interval")
    plt.plot(dates[:-1], new_cases[1, :], '-', alpha=0.2, label='New cases')
    plt.grid()
    plt.xlabel("Date")
    plt.ylabel("Incidence per 10,000")
    fig.autofmt_xdate()
    plt.show()
