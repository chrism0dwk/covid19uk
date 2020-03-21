"""Prediction functions"""
import optparse

import seaborn
import yaml
import pickle as pkl
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow_probability import stats as tfs
import matplotlib.pyplot as plt
import h5py

from covid.model import CovidUKODE
from covid.rdata import load_age_mixing, load_mobility_matrix, load_population
from covid.util import sanitise_settings, sanitise_parameter, seed_areas, doubling_time


def save_sims(sims, la_names, age_groups, filename):
    f = h5py.File(filename, 'w')
    dset_sim = f.create_dataset('prediction', data=sims)
    la_long = np.repeat(la_names, age_groups.shape[0]).astype(np.string_)
    age_long = np.tile(age_groups, la_names.shape[0]).astype(np.string_)
    dset_dims = f.create_dataset("dimnames", data=[b'sim_id', b't', b'state', b'la_age'])
    dset_la = f.create_dataset('la_names', data=la_long)
    dset_age = f.create_dataset('age_names', data=age_long)
    f.close()


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
                           np.datetime64('2020-09-01'), settings['holiday'], settings['bg_max_time'], 1)
    seeding = seed_areas(N, n_names)  # Seed 40-44 age group, 30 seeds by popn size
    state_init = simulator.create_initial_state(init_matrix=seeding)

    @tf.function
    def prediction(epsilon, beta, gamma):
        sims = tf.TensorArray(tf.float64, size=beta.shape[0])
        R0 = tf.TensorArray(tf.float64, size=beta.shape[0])
        for i in tf.range(beta.shape[0]):
            p = param
            p['epsilon'] = epsilon[i]
            p['beta1'] = beta[i]
            p['gamma'] = gamma[i]
            t, sim, solver_results = simulator.simulate(p, state_init)
            r = simulator.eval_R0(p)
            R0 = R0.write(i, r[0])
            sims = sims.write(i, sim)
        return sims.gather(range(beta.shape[0])), R0.gather(range(beta.shape[0]))

    draws = pi_beta.numpy()[np.arange(5000, pi_beta.shape[0], 10), :]
    with tf.device('/CPU:0'):
        sims, R0 = prediction(draws[:, 0], draws[:, 1], draws[:, 2])
        sims = tf.stack(sims) # shape=[n_sims, n_times, n_states, n_metapops]

        save_sims(sims, la_names, age_groups, 'pred_2020-03-15.h5')

        dub_time = [doubling_time(simulator.times, sim, '2020-03-01', '2020-04-01') for sim in sims.numpy()]

        # Sum over country
        sims = tf.reduce_sum(sims, axis=3)

        print("Plotting...", flush=True)
        dates = np.arange(date_range[0]-np.timedelta64(1, 'D'), np.datetime64('2020-09-01'),
                          np.timedelta64(1, 'D'))
        total_infected = tfs.percentile(tf.reduce_sum(sims[:, :, 1:3], axis=2), q=[2.5, 50, 97.5], axis=0)
        removed = tfs.percentile(sims[:, :, 3], q=[2.5, 50, 97.5], axis=0)
        removed_observed = tfs.percentile(removed * 0.1, q=[2.5, 50, 97.5], axis=0)

    fig = plt.figure()
    filler = plt.fill_between(dates, total_infected[0, :], total_infected[2, :], color='lightgray', alpha=0.8, label="95% credible interval")
    plt.fill_between(dates, removed[0, :], removed[2, :], color='lightgray', alpha=0.8)
    plt.fill_between(dates, removed_observed[0, :], removed_observed[2, :], color='lightgray', alpha=0.8)
    ti_line = plt.plot(dates, total_infected[1, :], '-', color='red', alpha=0.4, label="Infected")
    rem_line = plt.plot(dates, removed[1, :], '-', color='blue', label="Removed")
    ro_line = plt.plot(dates, removed_observed[1, :], '-', color='orange', label='Predicted detections')
    marks = plt.plot(data_dates, y, '+', label='Observed cases')
    plt.legend([ti_line[0], rem_line[0], ro_line[0], filler, marks[0]],
               ["Infected", "Removed", "Predicted detections", "95% credible interval", "Observed counts"])
    plt.grid(color='lightgray', linestyle='dotted')
    plt.xlabel("Date")
    plt.ylabel("Individuals")
    fig.autofmt_xdate()
    plt.show()

    # Number of new cases per day
    new_cases = tfs.percentile(removed[:, 1:] - removed[:, :-1],  q=[2.5, 50, 97.5], axis=0)/10000.
    fig = plt.figure()
    plt.fill_between(dates[:-1], new_cases[0, :], new_cases[2, :], color='lightgray', label="95% credible interval")
    plt.plot(dates[:-1], new_cases[1, :], '-', alpha=0.2, label='New cases')
    plt.grid(color='lightgray', linestyle='dotted')
    plt.xlabel("Date")
    plt.ylabel("Incidence per 10,000")
    fig.autofmt_xdate()
    plt.show()

    # R0
    R0_ci = tfs.percentile(R0, q=[2.5, 50, 97.5])
    print("R0:", R0_ci)
    fig = plt.figure()
    seaborn.kdeplot(R0.numpy(), ax=fig.gca())
    plt.title("R0")
    plt.show()

    # Doubling time
    dub_ci = tfs.percentile(dub_time, q=[2.5, 50, 97.5])
    print("Doubling time:", dub_ci)

    # Infectious period
    ip = tfs.percentile(1./pi_beta[3000:, 2], q=[2.5, 50, 97.5])
    print("Infectious period:", ip)
