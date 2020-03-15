import optparse
import yaml
import tensorflow as tf
import matplotlib.pyplot as plt

from covid.rdata import load_population, load_age_mixing, load_mobility_matrix
from covid.model import CovidUKODE
from covid.util import *


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
    def simulate():
        t0, sim_0, solve0 = model_term.simulate(param, state_init,
                                                settings['start'], settings['holiday'][0],
                                                settings['time_step'])
        t1, sim_1, solve1 = model_holiday.simulate(param, sim_0[-1, :, :],
                                                   settings['holiday'][0], settings['holiday'][1],
                                                   settings['time_step'], solver_state=None)
        t2, sim_2, _ = model_term.simulate(param, sim_1[-1, :, :],
                                           settings['holiday'][1], settings['end'],
                                           settings['time_step'], solver_state=None)
        t = tf.concat([t0, t1 + t0[-1], t2 + t0[-1] + t1[-1]], axis=0)
        sim = tf.concat([tf.expand_dims(state_init, axis=0), sim_0, sim_1, sim_2], axis=0)
        return t, sim

    t, sim = simulate()
    dates = settings['start'] + t.numpy().astype(np.timedelta64)
    print("Initial R0:", model_term.eval_R0(param))
    print("Doubling time:", doubling_time(dates, sim.numpy(), '2020-02-27','2020-03-13'))

    fig = plt.figure()
    removals = tf.reduce_sum(sim[:, 3, :], axis=1)
    infected = tf.reduce_sum(sim[:, 1:3, :], axis=[1,2])
    exposed = tf.reduce_sum(sim[:, 1, :], axis=1)
    date = np.squeeze(np.where(dates == np.datetime64('2020-03-13'))[0])
    print("Daily incidence 2020-03-13:", exposed[date])

    plt.plot(dates, removals[:-1]*0.10, '-', label='10% reporting')
    plt.plot(dates, infected[:-1], '-', color='red', label='Total infected')
    plt.plot(dates, removals[:-1], '-', color='gray', label='Total recovered/detected/died')

    plt.scatter(np.datetime64('2020-03-13'), 600, label='gov.uk cases 13th March 2020')
    plt.legend()
    plt.grid(True)
    fig.autofmt_xdate()
    plt.show()


