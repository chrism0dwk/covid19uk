import optparse
import time
import pickle as pkl

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

import numpy as np
import matplotlib.pyplot as plt
import yaml

from covid.model import CovidUKStochastic, load_data
from covid.util import sanitise_parameter, sanitise_settings, seed_areas

DTYPE = np.float64

def random_walk_mvnorm_fn(covariance, name=None):
    """Returns callable that adds Multivariate Normal noise to the input"""
    covariance = covariance + tf.eye(covariance.shape[0], dtype=tf.float64) * 1.e-9
    scale_tril = tf.linalg.cholesky(covariance)
    rv = tfp.distributions.MultivariateNormalTriL(loc=tf.zeros(covariance.shape[0], dtype=tf.float64),
                                                  scale_tril=scale_tril)

    def _fn(state_parts, seed):
        with tf.name_scope(name or 'random_walk_mvnorm_fn'):
            new_state_parts = [rv.sample() + state_part for state_part in state_parts]
            return new_state_parts

    return _fn


def sum_age_groups(sim):
    infec = sim[:, 2, :]
    infec = infec.reshape([infec.shape[0], 152, 17])
    infec_uk = infec.sum(axis=2)
    return infec_uk


def sum_la(sim):
    infec = sim[:, :, 2]
    infec = infec.reshape([infec.shape[0], 152, 17])
    infec_uk = infec.sum(axis=1)
    return infec_uk


def sum_total_removals(sim):
    remove = sim[:, 3, :]
    return remove.sum(axis=1)


def final_size(sim):
    remove = sim[:, :, 3]
    remove = remove.reshape([remove.shape[0], 152, 17])
    fs = remove[-1, :, :].sum(axis=0)
    return fs


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
    plt.grid()
    plt.legend()


def plot_infec_curve(ax, sim, label):
    infec_uk = sum_la(sim)
    infec_uk = infec_uk.sum(axis=1)
    times = np.datetime64('2020-02-20') + np.arange(infec_uk.shape[0])
    ax.plot(times, infec_uk, '-', label=label)


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


def plot_by_la(sim, labels, t0=np.datetime64('2020-02-20'), ax=None):
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
    Ns = N.reshape([152, 17]).sum(axis=0)
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

    parser = optparse.OptionParser()
    parser.add_option("--config", "-c", dest="config", default="ode_config.yaml",
                      help="configuration file")
    options, args = parser.parse_args()
    with open(options.config, 'r') as ymlfile:
        config = yaml.load(ymlfile)

    param = sanitise_parameter(config['parameter'])
    settings = sanitise_settings(config['settings'])

    data = load_data(config['data'], settings, DTYPE)

    model = CovidUKStochastic(M_tt=data['M_tt'],
                              M_hh=data['M_hh'],
                              C=data['C'],
                              N=data['pop']['n'].to_numpy(),
                              W=data['W'],
                              date_range=settings['inference_period'],
                              holidays=settings['holiday'],
                              lockdown=settings['lockdown'],
                              time_step=1.)

    seeding = seed_areas(data['pop']['n'].to_numpy(), data['pop']['Area.name.2'])  # Seed 40-44 age group, 30 seeds by popn size
    state_init = model.create_initial_state(init_matrix=seeding)

    start = time.perf_counter()
    t, sim = model.simulate(param, state_init)
    end = time.perf_counter()
    print(f'Run 1 Complete in {end - start} seconds')

    start = time.perf_counter()
    for i in range(1):
        t, upd = model.simulate(param, state_init)
    end = time.perf_counter()
    print(f'Run 2 Complete in {(end - start)/1.} seconds')

    # Plotting functions
    fig_attack = plt.figure()
    fig_uk = plt.figure()
    sim = tf.reduce_sum(upd, axis=-2)

    plot_age_attack_rate(fig_attack.gca(), sim, data['pop']['n'].to_numpy(), "Attack Rate")
    fig_attack.suptitle("Attack Rate")
    plot_infec_curve(fig_uk.gca(), sim.numpy(), "Infections")
    fig_uk.suptitle("UK Infections")

    fig_attack.autofmt_xdate()
    fig_uk.autofmt_xdate()
    fig_attack.gca().grid(True)
    fig_uk.gca().grid(True)
    plt.show()

    with open('stochastic_sim.pkl', 'wb') as f:
        pkl.dump({'events': upd.numpy(), 'state_init': state_init.numpy()}, f)
