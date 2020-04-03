import optparse
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml

from covid.model import CovidUKODE
from covid.model import load_data
from covid.util import sanitise_parameter, sanitise_settings, seed_areas


def sum_age_groups(sim):
    infec = sim[:, :, 2]
    infec = infec.reshape([infec.shape[0], 152, 17])
    infec_uk = infec.sum(axis=2)
    return infec_uk


def sum_la(sim):
    infec = sim[:, :, 2]
    infec = infec.reshape([infec.shape[0], 149, 17])
    infec_uk = infec.sum(axis=1)
    return infec_uk


def sum_total_removals(sim):
    remove = sim[:, :, 3]
    return remove.sum(axis=1)


def final_size(sim):
    remove = sim[:, :, 3]
    remove = remove.reshape([remove.shape[0], 149, 17])
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
    infec_uk = infec_uk.sum(axis=-1)
    removals = sum_total_removals(sim)
    times = np.datetime64('2020-02-20') + np.arange(removals.shape[0])
    plt.plot(times, infec_uk, 'r-', label='Infected')
    plt.plot(times, removals, 'b-', label='Removed')
    plt.title('UK total cases')
    plt.xlabel('Date')
    plt.ylabel('Num infected or removed')
    plt.grid()
    plt.legend()


def plot_infec_curve(ax, sim, label):
    infec_uk = sum_la(sim)
    infec_uk = infec_uk.sum(axis=-1)
    times = np.datetime64('2020-02-20') + np.arange(infec_uk.shape[0])
    ax.plot(times, infec_uk, '-', label=label)


def plot_by_age(sim, labels, t0=np.datetime64('2020-02-20'), ax=None):
    if ax is None:
        ax = plt.figure().gca()
    infec_uk = sum_la(sim)
    total_uk = infec_uk.mean(axis=-1)
    t = t0 + np.arange(infec_uk.shape[0])
    colours = plt.cm.viridis(np.linspace(0., 1., infec_uk.shape[1]))
    for i in range(infec_uk.shape[1]):
        ax.plot(t, infec_uk[:, i], 'r-', alpha=0.4, color=colours[i], label=labels[i])
    ax.plot(t, total_uk, '-', color='black', label='Mean')
    return ax


def plot_by_la(sim, labels, t0=np.datetime64('2020-02-20'), ax=None):
    if ax is None:
        ax = plt.figure().gca()
    infec_uk = sum_age_groups(sim)
    total_uk = infec_uk.mean(axis=-1)
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
    print("Overall attack rate: ", np.sum(fs) / np.sum(N))

    # Total UK epidemic curve
    plot_total_curve(sim)
    plt.xticks(rotation=45, horizontalalignment="right")
    plt.savefig('total_uk_curve.pdf')
    plt.show()

    # TotalUK epidemic curve by age-group
    fig, ax = plt.subplots(1, 2, figsize=[24, 12])
    plot_by_la(sim, data['la_names'], ax=ax[0])
    plot_by_age(sim, data['age_groups'], ax=ax[1])
    ax[1].legend()
    plt.xticks(rotation=45, horizontalalignment="right")
    fig.autofmt_xdate()
    plt.savefig('la_age_infec_curves.pdf')
    plt.show()

    # Plot attack rate
    plt.figure(figsize=[4, 2])
    plt.plot(data['age_groups'], attack_rate, 'o-')
    plt.xticks(rotation=90)
    plt.title('Age-specific attack rate')
    plt.savefig('age_attack_rate.pdf')
    plt.show()


def doubling_time(t, sim, t1, t2):
    t1 = np.where(t == np.datetime64(t1))[0]
    t2 = np.where(t == np.datetime64(t2))[0]
    delta = t2 - t1
    r = sum_total_removals(sim)
    q1 = r[t1]
    q2 = r[t2]
    return delta * np.log(2) / np.log(q2 / q1)


def plot_age_attack_rate(ax, sim, N, label):

    Ns = N.sum(level=1)  # Sum over ages
    fs = final_size(sim.numpy())
    attack_rate = fs / Ns
    ax.plot(data['age_groups'], attack_rate, 'o-', label=label)


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

    model = CovidUKODE(M_tt=data['M_tt'],
                       M_hh=data['M_hh'],
                       C=data['C'],
                       N=data['pop']['n'].to_numpy(),
                       W=data['W'].to_numpy(),
                       date_range=settings['prediction_period'],
                       holidays=settings['holiday'],
                       lockdown=settings['lockdown'],
                       time_step=1)

    seeding = seed_areas(data['pop']['n'])  # Seed 40-44 age group, 30 seeds by popn size
    state_init = model.create_initial_state(init_matrix=seeding)

    print('R0_term=', model.eval_R0(param))

    @tf.function(autograph=False, experimental_compile=True)
    def compiled_sim():
        return model.simulate(param, state_init)

    start = time.perf_counter()
    t, sim, _ = compiled_sim()
    end = time.perf_counter()
    print(f'Complete in {end - start} seconds')

    dates = np.arange(settings['prediction_period'][0], settings['prediction_period'][1],
                      np.timedelta64(1, 'D'))
    dt = doubling_time(dates, sim.numpy(), '2020-03-01', '2020-03-31')
    print(f"Doubling time: {dt}")


    fig_attack = plt.figure()
    fig_uk = plt.figure()

    plot_age_attack_rate(fig_attack.gca(), sim, data['pop']['n'], "Attack Rate")
    fig_attack.suptitle("Attack Rate")
    plot_infec_curve(fig_uk.gca(), sim.numpy(), "Infections")
    fig_uk.suptitle("UK Infections")

    fig_attack.autofmt_xdate()
    fig_uk.autofmt_xdate()
    fig_attack.gca().grid(True)
    fig_uk.gca().grid(True)
    plt.show()

