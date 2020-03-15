"Covid analysis utility functions"

import numpy as np

def sanitise_parameter(par_dict):
    """Sanitises a dictionary of parameters"""
    par = ['beta1', 'beta2', 'nu', 'gamma']
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


def sum_total_removals(sim):
    remove = sim[:, 3, :]
    return remove.sum(axis=1)


def doubling_time(t, sim, t1, t2):
    t1 = np.where(t == np.datetime64(t1))[0]
    t2 = np.where(t == np.datetime64(t2))[0]
    delta = t2 - t1
    r = sum_total_removals(sim)
    q1 = r[t1]
    q2 = r[t2]
    return delta * np.log(2) / np.log(q2 / q1)


def final_size(sim):
    remove = sim[:, 3, :]
    remove = remove.reshape([remove.shape[0], 152, 17])
    fs = remove[-1, :, :].sum(axis=0)
    return fs

