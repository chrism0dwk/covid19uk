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

    case_timeseries = phe_linelist_timeseries(config['data']['reported_cases'])
    y = zero_cases(case_timeseries, data['pop'])
    y = y[settings['inference_period'][0]:settings['inference_period'][1]]

    date_range = settings['inference_period']

    with open(config['output']['posterior'], 'rb') as f:
        pi_beta = pkl.load(f)

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

    seeding = seed_areas(data['pop']['n'])
    state_init = simulator.create_initial_state(init_matrix=seeding)

    @tf.function
    def prediction(beta, beta3, gamma, I0, r_):
        sims = tf.TensorArray(tf.float64, size=beta.shape[0])
        Rt = tf.TensorArray(tf.float64, size=beta.shape[0])
        for i in tf.range(beta.shape[0]):
            p = param
            p['beta1'] = beta[i]
            p['beta3'] = beta3[i]
            p['gamma'] = gamma[i]
            p['I0'] = I0[i]
            p['r'] = r_[i]
            state_init = simulator.create_initial_state(seeding * p['I0'])
            t, sim, solver_results = simulator.simulate(p, state_init)
            r0 = simulator.eval_Rt(p, [t[0]], sim[0, :, 0])  # Todo: Reduce memory usage in batching,
                                                              #   currently limited to first 10 timesteps.
            r1 = simulator.eval_Rt(p, [t[29]], sim[29, :, 0])
            r2 = simulator.eval_Rt(p, [t[42]], sim[42, :, 0])
            Rt = Rt.write(i, [r0, r1, r2])
            sims = sims.write(i, sim)
        return sims.gather(range(beta.shape[0])), Rt.gather(range(beta.shape[0]))

    draws = pi_beta.numpy()[np.arange(5000, pi_beta.shape[0], 60), :]
    with tf.device('/CPU:0'):  # Todo: Using CPU because GPU goes OOM
        sims, r0 = prediction(draws[:, 0], draws[:, 1], draws[:, 2], draws[:, 3], draws[:, 4])
        sims = tf.stack(sims)  # shape=[n_sims, n_times, n_metapops, n_states]
        save_sims(simulator.times, sims, data['la_names'], data['age_groups'], config['output']['prediction'])
        dub_time = [doubling_time(simulator.times, sim, '2020-03-01', '2020-04-01') for sim in sims.numpy()]

        fig, ax = plot_prediction(settings['prediction_period'], sims, y.sum(level=0))
        ax.vlines(settings['lockdown'][0], *ax.get_ylim(), colors='gray', linestyles='dashed', label='Lockdown')
        plt.legend()
        plt.show()
        fig, ax = plot_case_incidence(settings['prediction_period'], sims.numpy())
        ax.set_ylim([-11.327, 119.227])
        ax.vlines(settings['lockdown'][0], *ax.get_ylim(), colors='gray', linestyles='dashed', label='Lockdown')
        plt.legend()
        plt.show()


    # Plot Rt for select dates as above
    r0 = tf.squeeze(r0).numpy()
    R0_ci = np.percentile(r0, q=[2.5, 50, 97.5], axis=0)
    print("Rt:", R0_ci)
    fig, ax = plt.subplots(1, 3)
    rt_dates = ['2020-02-19', '2020-03-19', '2020-04-01']
    for i, date in enumerate(rt_dates):
        seaborn.kdeplot(r0[:, i], ax=ax[i])
        ax[i].set_xlabel(f"Rt({date})")
    plt.title("R0")
    plt.show()
    np.savetxt(config['output']['rt'], r0, header='.'.join(rt_dates))

    # Doubling time
    dub_ci = tfs.percentile(dub_time, q=[2.5, 50, 97.5])
    print("Doubling time:", dub_ci)

    # Infectious period
    ip = tfs.percentile(1./pi_beta[3000:, 1], q=[2.5, 50, 97.5])
    print("Infectious period:", ip)
