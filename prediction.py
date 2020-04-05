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

from covid.model import CovidUKODE, load_data
from covid.pydata import collapse_pop, phe_linelist_timeseries, zero_cases
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

    data = load_data(config['data'], settings)

    # case_reports = pd.read_csv(config['data']['reported_cases'])
    # case_reports['DateVal'] = pd.to_datetime(case_reports['DateVal'])
    # case_reports = case_reports[case_reports['DateVal'] >= settings['inference_period'][0]]
    # date_range = [case_reports['DateVal'].min(), case_reports['DateVal'].max()]
    # y = case_reports['CMODateCount'].to_numpy()[1:]
    # #y_incr = y[1:] - y[-1:]

    case_timeseries = phe_linelist_timeseries(config['data']['reported_cases'])
    y = zero_cases(case_timeseries, data['pop'])


    date_range = [y.index.levels[0].min(), y.index.levels[0].max()]

    with open('pi_beta_2020-03-29.pkl', 'rb') as f:
        pi_beta = pkl.load(f)

    # Predictive distribution of epidemic spread
    data_dates = np.arange(date_range[0],
                           date_range[1]+np.timedelta64(1,'D'),
                           np.timedelta64(1, 'D'))

    simulator = CovidUKODE(M_tt=data['M_tt'],
                           M_hh=data['M_hh'],
                           C=data['C'],
                           N=data['pop']['n'].to_numpy(),
                           W=data['W'].to_numpy(),
                           date_range=[settings['prediction_period'][0],
                                       settings['prediction_period'][1]-np.timedelta64(1, 'D')],
                           holidays=settings['holiday'],
                           lockdown=settings['lockdown'],
                           time_step=1)
    seeding = y[:'2020-03-09'].sum(level=[1 , 2]).to_numpy()# seed_areas(data['pop']['n'])  # Seed 40-44 age group, 30 seeds by popn size
    state_init = simulator.create_initial_state(init_matrix=seeding)

    @tf.function
    def prediction(beta, beta3, gamma, I0, r_):
        sims = tf.TensorArray(tf.float64, size=beta.shape[0])
        R0 = tf.TensorArray(tf.float64, size=beta.shape[0])
        for i in tf.range(beta.shape[0]):
            p = param
            p['beta1'] = beta[i]
            p['beta3'] = beta3[i]
            p['gamma'] = gamma[i]
            p['I0'] = I0[i]
            p['r'] = r_[i]
            state_init = simulator.create_initial_state(seeding * p['I0'])
            t, sim, solver_results = simulator.simulate(p, state_init)
            r = simulator.eval_R0(p)
            R0 = R0.write(i, r[0])
            sims = sims.write(i, sim)
        return sims.gather(range(beta.shape[0])), R0.gather(range(beta.shape[0]))

    draws = pi_beta.numpy()[np.arange(5000, pi_beta.shape[0], 60), :]
    with tf.device('/CPU:0'):
        sims, R0 = prediction(draws[:, 0], draws[:, 1], draws[:, 2], draws[:, 3], draws[:, 4])
        sims = tf.stack(sims)  # shape=[n_sims, n_times, n_metapops, n_states]
        save_sims(simulator.times, sims, data['la_names'], data['age_groups'], 'pred_2020-03-29.h5')
        dub_time = [doubling_time(simulator.times, sim, '2020-03-01', '2020-04-01') for sim in sims.numpy()]

        plot_prediction(settings['prediction_period'], sims, y.sum(level=0))
        plot_case_incidence(settings['prediction_period'], sims.numpy())

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
