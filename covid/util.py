"Covid analysis utility functions"

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import h5py

tfs = tfp.stats

def sanitise_parameter(par_dict):
    """Sanitises a dictionary of parameters"""
    d = {key: np.float64(val) for key, val in par_dict.items()}
    return d


def sanitise_settings(par_dict):
    d = {'inference_period': np.array(par_dict['inference_period'], dtype=np.datetime64),
         'prediction_period': np.array(par_dict['prediction_period'], dtype=np.datetime64),
         'time_step': float(par_dict['time_step']),
         'holiday': np.array([np.datetime64(date) for date in par_dict['holiday']]),
         'lockdown': np.array([np.datetime64(date) for date in par_dict['lockdown']])}
    return d


def seed_areas(N, names, age_group=8, num_la=152, num_age=17, n_seed=30.):
    areas = ['Inner London',
             'Outer London',
             'West Midlands (Met County)',
             'Greater Manchester (Met County)']

    names_matrix = names.to_numpy().reshape([num_la, num_age])

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


def brick_to_imperial_csv(creation_date, date, sim, required_dates=None):
    """Converts a simulation brick to an Imperial-style csv
    :param creation_date: date timestamp
    :param date: the timeseries date
    :param sim: the 4D array containing the prediction [M, T, S, L] where M is number of simulations,
    T is number of time points, S is number of states, L is number of age/LA combinations.
    :param required_dates: the dates between which the forecast is required.  Semi-closed interval [low, high)

    :returns a Pandas DataFrame object
    """

    sim = sim.sum(axis=3)  #  Todo: sum age/la for now
    date = np.array(date, dtype=np.datetime64)

    if required_dates is not None:
        required_dates = np.array(required_dates, dtype=np.datetime64)
    else:
        required_dates = [date.min(), date.max()]

    # Removals
    cases = sim[:, :, 3]
    cases_mean = cases.mean(axis=0)
    cases_q = np.percentile(a=cases, q=[2.5, 97.5], axis=0)

    rv = pd.DataFrame({'Group': 'Lancaster',
                       'Scenario': 'Forecast',
                       'CreationDate': creation_date,
                       'DateOfValue': date,
                       'Geography': 'England',
                       'ValueType': 'CumCases',
                       'Value': cases_mean,
                       'LowerBound': cases_q[0],
                       'UpperBound': cases_q[1]
                       })

    rv = rv[np.logical_and(rv['DateOfValue'] >= required_dates[0],
                           rv['DateOfValue'] < required_dates[1])]

    return rv


def save_sims(dates, sims, la_names, age_groups, filename):
    f = h5py.File(filename, 'w')
    dset_sim = f.create_dataset('prediction', data=sims, compression='gzip', compression_opts=4)
    la_long = np.repeat(la_names, age_groups.shape[0]).astype(np.string_)
    age_long = np.tile(age_groups, la_names.shape[0]).astype(np.string_)
    dset_dims = f.create_dataset("dimnames", data=[b'sim_id', b't', b'la_age', b'state'])
    dset_la = f.create_dataset('la_names', data=la_long)
    dset_age = f.create_dataset('age_names', data=age_long)
    dset_times = f.create_dataset('date', data=dates.astype(np.string_))
    f.close()


def extract_locs(in_file: str, out_file: str, loc: list):


    f = h5py.File(in_file, 'r')
    la_names = f['la_names'][:].astype(str)
    la_loc = np.isin(la_names, loc)

    extract = f['prediction'][:, :, la_loc, :]

    save_sims(f['date'][:], extract, f['la_names'][la_loc],
              f['age_names'][la_loc], out_file)
    f.close()
    return extract


def extract_liverpool(in_file: str, out_file: str):
    la_seq = [
        "E06000006",
        "E06000007",
        "E06000008",
        "E06000009",
        "E08000011",  # Knowsley
        "E08000012",  # Liverpool
        "E08000013",  # St Helens
        "E08000014",  # Sefton
        "E08000015"  # Wirral
    ]
    extract_locs(in_file, out_file, la_seq)
