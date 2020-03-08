import optparse
import time

import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import yaml

from covid.model import CovidUKODE
from covid.rdata import *


def sanitise_parameter(par_dict):
    """Sanitises a dictionary of parameters"""
    par = ['beta1', 'beta2', 'nu','gamma']
    d = {key: np.float64(par_dict[key]) for key in par}
    return d


def sanitise_settings(par_dict):
    d = {'start': np.datetime64(par_dict['start']),
         'end': np.datetime64(par_dict['end']),
         'time_step': float(par_dict['time_step']),
         'holiday': np.array([np.datetime64(date) for date in par_dict['holiday']])}
    return d


def seed_areas(N, names, age_group=8, num_la=152, num_age=17, n_seed=30.):
    areas = ['Inner London',
             'Outer London',
             'West Midlands (Met County)',
             'Greater Manchester (Met County)']

    names_matrix = names['Area.name.2'].to_numpy().reshape([num_la, num_age])

    seed_areas = np.in1d(names_matrix[:, age_group], areas)
    N_matrix = N.reshape([num_la, num_age])  # LA x Age

    pop_size_sub = N_matrix[seed_areas, age_group]  # Gather
    n = np.round(n_seed * pop_size_sub / pop_size_sub.sum())

    seeding = np.zeros_like(N_matrix)
    seeding[seed_areas, age_group] = n  # Scatter
    return seeding


def sum_age_groups(sim):
    infec = sim[:, 2, :]
    infec = infec.reshape([infec.shape[0], 152, 17])
    infec_uk = infec.sum(axis=2)
    return infec_uk


def sum_la(sim):
    infec = sim[:, 2, :]
    infec = infec.reshape([infec.shape[0], 152, 17])
    infec_uk = infec.sum(axis=1)
    return infec_uk


def sum_total_removals(sim):
    remove = sim[:, 3, :]
    return remove.sum(axis=1)


def final_size(sim):
    remove = sim[:, 3, :]
    remove = remove.reshape([remove.shape[0], 152, 17])
    fs = remove[-1, :, :].sum(axis=0)
    return fs


def write_hdf5(filename, param, t, sim):
    with h5py.File(filename, "w") as f:
        dset_sim = f.create_dataset("simulation", sim.shape, dtype='f')
        dset_sim[:] = sim
        dset_t = f.create_dataset("time", t.shape, dtype='f')
        dset_t[:] = t
        grp_param = f.create_group("parameter")
        for k, v in param.items():
            d_beta = grp_param.create_dataset(k, [1], dtype='f')
            d_beta[()] = v



def plot_total_curve(sim):
    infec_uk = sum_la(sim)
    infec_uk = infec_uk.sum(axis=1)
    removals = sum_total_removals(sim)
    times = np.datetime64('2020-02-20') + np.arange(removals.shape[0])
    plt.plot(times, infec_uk, 'r-', label='Infected')
    plt.plot(times, removals, 'b-', label='Removed')
    plt.title('UK total cases')
    plt.xlabel('Date')
    plt.ylabel('Num infected or removed')
    plt.legend()


def plot_by_age(sim, labels, t0=np.datetime64('2020-02-20'), ax=None):
    if ax is None:
        ax = plt.figure().gca()
    infec_uk = sum_la(sim)
    total_uk = infec_uk.mean(axis=1)
    t = t0 + np.arange(infec_uk.shape[0])
    colours = plt.cm.viridis(np.linspace(0., 1., infec_uk.shape[1]))
    for i in range(infec_uk.shape[1]):
        ax.plot(t, infec_uk[:, i], 'r-', alpha=0.4, color=colours[i], label=labels[i])
    ax.plot(t, total_uk, '-', color='black', label='Mean')
    return ax


def plot_by_la(sim, labels,t0=np.datetime64('2020-02-20'), ax=None):
    if ax is None:
        ax = plt.figure().gca()
    infec_uk = sum_age_groups(sim)
    total_uk = infec_uk.mean(axis=1)
    t = t0 + np.arange(infec_uk.shape[0])
    colours = plt.cm.viridis(np.linspace(0., 1., infec_uk.shape[1]))
    for i in range(infec_uk.shape[1]):
        ax.plot(t, infec_uk[:, i], 'r-', alpha=0.4, color=colours[i], label=labels[i])
    ax.plot(t, total_uk, '-', color='black', label='Mean')
    return ax


def draw_figs(sim, N):
    # Attack rate
    N = N.reshape([152, 17]).sum(axis=0)
    fs = final_size(sim)
    attack_rate = fs / N
    print("Attack rate:", attack_rate)
    print("Overall attack rate: ", np.sum(fs)/np.sum(N))

    # Total UK epidemic curve
    plot_total_curve(sim)
    plt.xticks(rotation=45, horizontalalignment="right")
    plt.savefig('total_uk_curve.pdf')
    plt.show()

    # TotalUK epidemic curve by age-group
    fig, ax = plt.subplots(1, 2, figsize=[24, 12])
    plot_by_la(sim, la_names, ax=ax[0])
    plot_by_age(sim, age_groups, ax=ax[1])
    ax[1].legend()
    plt.xticks(rotation=45, horizontalalignment="right")
    fig.autofmt_xdate()
    plt.savefig('la_age_infec_curves.pdf')
    plt.show()

    # Plot attack rate
    plt.figure(figsize=[4, 2])
    plt.plot(age_groups, attack_rate, 'o-')
    plt.xticks(rotation=90)
    plt.title('Age-specific attack rate')
    plt.savefig('age_attack_rate.pdf')
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


# Straight, no school closures
    model_term = CovidUKODE(K_tt, T, N)
    model_holiday = CovidUKODE(K_hh/2., T/2., N)

    seeding = seed_areas(N, n_names)  # Seed 40-44 age group, 30 seeds by popn size
    state_init = model_term.create_initial_state(init_matrix=seeding)

    print('R_term=', model_term.eval_R0(param))
    print('R_holiday=', model_holiday.eval_R0(param))

    # School holidays and closures
    @tf.function
    def simulate():
        t0, sim_0 = model_term.simulate(param, state_init,
                                        settings['start'], settings['holiday'][0],
                                        settings['time_step'])
        t1, sim_1 = model_holiday.simulate(param, sim_0[-1, :, :],
                                           settings['holiday'][0], settings['holiday'][1],
                                           settings['time_step'])
        t2, sim_2 = model_term.simulate(param, sim_1[-1, :, :],
                                        settings['holiday'][1], settings['end'],
                                        settings['time_step'])
        t = tf.concat([t0, t1, t2], axis=0)
        sim = tf.concat([tf.expand_dims(state_init, axis=0), sim_0, sim_1, sim_2], axis=0)
        return t, sim

    start = time.perf_counter()
    t, sim = simulate()
    end = time.perf_counter()
    print(f'Complete in {end-start} seconds')

    draw_figs(sim.numpy(), N)

    if 'simulation' in config['output']:
        write_hdf5(config['output']['simulation'], param, t, sim)

    print(f"Complete in {end-start} seconds")