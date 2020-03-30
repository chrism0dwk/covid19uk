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
from covid.pydata import load_commute_volume
from covid.util import sanitise_settings, sanitise_parameter, seed_areas, doubling_time, save_sims
from covid.plotting import plot_prediction, plot_case_incidence

DTYPE = np.float64



if __name__ == '__main__':

    parser = optparse.OptionParser()
    parser.add_option("--config", "-c", dest="config", default="ode_config.yaml",
                      help="configuration file")
    options, args = parser.parse_args()
    with open(options.config, 'r') as ymlfile:
        config = yaml.load(ymlfile)

    param = sanitise_parameter(config['parameter'])
    settings = sanitise_settings(config['settings'])

    W = load_commute_volume(config['data']['commute_volume'], settings['prediction_period'])

    K_tt, age_groups = load_age_mixing(config['data']['age_mixing_matrix_term'])
    K_hh, _ = load_age_mixing(config['data']['age_mixing_matrix_hol'])

    T, la_names = load_mobility_matrix(config['data']['mobility_matrix'])
    np.fill_diagonal(T, 0.)

    N, n_names = load_population(config['data']['population_size'])

    K_tt = K_tt.astype(DTYPE)
    K_hh = K_hh.astype(DTYPE)
    T = T.astype(DTYPE)
    N = N.astype(DTYPE)
    W = W.to_numpy().astype(DTYPE)

    case_reports = pd.read_csv(config['data']['reported_cases'])
    case_reports['DateVal'] = pd.to_datetime(case_reports['DateVal'])
    case_reports = case_reports[case_reports['DateVal'] >= settings['inference_period'][0]]
    date_range = [case_reports['DateVal'].min(), case_reports['DateVal'].max()]
    y = case_reports['CumCases'].to_numpy()
    y_incr = y[1:] - y[-1:]

    with open('pi_beta_2020-03-15.pkl', 'rb') as f:
        pi_beta = pkl.load(f)

    # Predictive distribution of epidemic spread
    data_dates = np.arange(date_range[0],
                           date_range[1]+np.timedelta64(1,'D'),
                           np.timedelta64(1, 'D'))
    simulator = CovidUKODE(M_tt=K_tt,
                           M_hh=K_hh,
                           C=T,
                           W=W,
                           N=N,
                           date_range=[settings['prediction_period'][0],
                                       settings['prediction_period'][1]],
                           holidays=settings['holiday'],
                           time_step=1)
    seeding = seed_areas(N, n_names)  # Seed 40-44 age group, 30 seeds by popn size
    state_init = simulator.create_initial_state(init_matrix=seeding)

    @tf.function
    def prediction(beta, gamma, I0, r_):
        sims = tf.TensorArray(tf.float64, size=beta.shape[0])
        R0 = tf.TensorArray(tf.float64, size=beta.shape[0])
        for i in tf.range(beta.shape[0]):
            p = param
            p['beta1'] = beta[i]
            p['gamma'] = gamma[i]
            p['I0'] = I0[i]
            p['r'] = r_[i]
            t, sim, solver_results = simulator.simulate(p, state_init)
            r = simulator.eval_R0(p)
            R0 = R0.write(i, r[0])
            sims = sims.write(i, sim)
        return sims.gather(range(beta.shape[0])), R0.gather(range(beta.shape[0]))

    draws = pi_beta.numpy()[np.arange(5000, pi_beta.shape[0], 30), :]
    with tf.device('/CPU:0'):
        sims, R0 = prediction(draws[:, 0], draws[:, 1], draws[:, 2], draws[:, 3])
        sims = tf.stack(sims)  # shape=[n_sims, n_times, n_metapops, n_states]
        save_sims(simulator.times, sims, la_names, age_groups, 'pred_2020-03-23.h5')
        dub_time = [doubling_time(simulator.times, sim, '2020-03-01', '2020-04-01') for sim in sims.numpy()]


    plot_prediction(settings['prediction_period'], sims, case_reports)
    plot_case_incidence(settings['prediction_period'], sims)


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
    ip = tfs.percentile(1./pi_beta[3000:, 1], q=[2.5, 50, 97.5])
    print("Infectious period:", ip)
