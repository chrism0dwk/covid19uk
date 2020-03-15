"""Code implementing WHI restrictions"""

import optparse
import yaml
import numpy as np
from scipy import optimize as opt
import tensorflow as tf
import matplotlib.pyplot as plt

from covid.rdata import load_population, load_age_mixing, load_mobility_matrix
from covid.model import CovidUKODE
from covid.util import final_size
from covid.util import *


def optimise_beta1(R0, model, param):

    def opt_fn(beta, R0):
        param['beta1'] = beta
        r0, i = model.eval_R0(param, 1e-9)
        return (r0 - R0)**2

    res = opt.minimize_scalar(opt_fn, args=R0)
    return res['x']



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

    model_term = CovidUKODE(K_tt, T, N)
    model_holiday = CovidUKODE(K_hh, T, N)

    seeding = seed_areas(N, n_names)  # Seed 40-44 age group, 30 seeds by popn size
    state_init = model_term.create_initial_state(init_matrix=seeding)

    @tf.function()
    def simulate(param):
        t0, sim_0, solve0 = model_term.simulate(param, state_init,
                                                np.datetime64('2020-02-04'), settings['holiday'][0],
                                                settings['time_step'])
        t1, sim_1, solve1 = model_holiday.simulate(param, sim_0[-1, :, :],
                                                   settings['holiday'][0], settings['holiday'][1],
                                                   settings['time_step'], solver_state=None)
        t2, sim_2, _ = model_term.simulate(param, sim_1[-1, :, :],
                                           settings['holiday'][1], np.datetime64('2021-12-01'),
                                           settings['time_step'], solver_state=None)
        t = tf.concat([t0, t1 + t0[-1], t2 + t0[-1] + t1[-1]], axis=0)
        sim = tf.concat([tf.expand_dims(state_init, axis=0), sim_0, sim_1, sim_2], axis=0)
        return t, sim

    t, sim = simulate(param)
    dates = settings['start'] + t.numpy().astype(np.timedelta64)
    print("Initial R0:", model_term.eval_R0(param))
    print("Doubling time:", doubling_time(dates, sim.numpy(), '2020-02-27','2020-03-13'))
    print("Attack rate:", np.sum(final_size(sim.numpy()))/np.sum(N))

    param['beta1'] = optimise_beta1(1.6, model_term, param)
    t_whi, sim_whi = simulate(param)
    print("Initial R0:", model_term.eval_R0(param))
    print("Doubling time:", doubling_time(dates, sim.numpy(), '2020-02-27', '2020-03-13'))
    print("Attack rate:", np.sum(final_size(sim_whi.numpy()))/np.sum(N))

    param['beta1'] = optimise_beta1(2.2, model_term, param)
    t_whi, sim_whi_05 = simulate(param)
    print("Initial R0:", model_term.eval_R0(param))
    print("Doubling time:", doubling_time(dates, sim.numpy(), '2020-02-27', '2020-03-13'))
    print("Attack rate:", np.sum(final_size(sim_whi_05.numpy()))/np.sum(N))
    fig = plt.figure()
    removals = tf.reduce_sum(sim[:, 3, :], axis=1)
    infected = tf.reduce_sum(sim[:, 1:3, :], axis=[1, 2])
    infected_whi = tf.reduce_sum(sim_whi[:, 1:3, :], axis=[1, 2])
    infected_whi_05 = tf.reduce_sum(sim_whi_05[:, 1:3, :], axis=[1, 2])

    date = np.squeeze(np.where(dates == np.datetime64('2020-03-13'))[0])
    plt.plot(dates, infected[:-1]/1e6, '-', color='lightblue', label='$\eta=0, R_\star=2.7$')
    plt.plot(dates, infected_whi[:-1]/1e6, '-', color='pink', label='$\eta=1, R_\star=1.6$')
    plt.plot(dates, infected_whi_05[:-1]/1e6, '-', color='orange', label='$\eta=0.5, R_\star=2.2$')
    plt.legend()
    plt.ylabel("Cases in $10^6$ individuals")
    plt.xlabel("Date")
    plt.grid(True)
    fig.autofmt_xdate()
    plt.show()


