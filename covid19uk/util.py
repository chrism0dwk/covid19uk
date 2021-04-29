"""Covid analysis utility functions"""

import yaml
import numpy as np
import pandas as pd
import h5py
import xarray
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import dtype_util

tfd = tfp.distributions
tfs = tfp.stats


def copy_nc_attrs(src, dest):
    """Copies dataset attributes between two NetCDF datasets"""
    with xarray.open_dataset(src) as s:
        attrs = s.attrs
    # Write empty root dataset with attributes
    ds = xarray.Dataset(attrs=attrs)
    ds.to_netcdf(dest, mode="a")


def load_config(config_filename):
    with open(config_filename, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def sanitise_parameter(par_dict):
    """Sanitises a dictionary of parameters"""
    d = {key: np.float64(val) for key, val in par_dict.items()}
    return d


def sanitise_settings(par_dict):
    d = {
        "inference_period": np.array(
            par_dict["inference_period"], dtype=np.datetime64
        ),
        "prediction_period": np.array(
            par_dict["prediction_period"], dtype=np.datetime64
        ),
        "time_step": float(par_dict["time_step"]),
        "holiday": np.array(
            [np.datetime64(date) for date in par_dict["holiday"]]
        ),
        "lockdown": np.array(
            [np.datetime64(date) for date in par_dict["lockdown"]]
        ),
    }
    return d


def seed_areas(N, age_group=40, n_seed=30.0):
    inner_london = [
        "E09000001,E09000033",  # City of London, City of Westminster
        "E09000007",  # Camden
        "E09000012",  # Hackney
        "E09000013",  # Hammersmith and Fulham
        "E09000014",  # Haringay
        "E09000019",  # Islington
        "E09000020",  # Kensignton and Chelsea
        "E09000022",  # Lambeth
        "E09000023",  # Lewisham
        "E09000025",  # Newham
        "E09000028",  # Southwark
        "E09000030",  # Tower Hamlets
        "E09000032",
    ]  # Wandsworth
    outer_london = [
        "E09000002",  # Barking and Dagenham
        "E09000003",  # Barnet
        "E09000004",  # Bexley
        "E09000005",  # Brent
        "E09000006",  # Bromley
        "E09000008",  # Croydon
        "E09000009",  # Ealing
        "E09000010",  # Enfield
        "E09000011",  # Greenwich
        "E09000015",  # Harrow
        "E09000016",  # Havering
        "E09000017",  # Hillingdon
        "E09000018",  # Hounslow
        "E09000021",  # Kingston upon Thames
        "E09000024",  # Merton
        "E09000026",  # Redbridge
        "E09000027",  # Richmond upon Thames
        "E09000029",  # Sutton
        "E09000031",
    ]  # Waltham Forest

    west_midlands = [
        "E08000025",  # Birmingham
        "E08000026",  # Coventry
        "E08000029",  # Solihull
        "E08000028",  # Sandwell
        "E08000030",  # Walsall
        "E08000027",  # Dudley
        "E08000031",
    ]  # Wolverhampton
    greater_manchester = [
        "E08000001",  # Bolton
        "E08000002",  # Bury
        "E08000003",  # Manchester
        "E08000004",  # Oldham
        "E08000005",  # Rochdale
        "E08000006",  # Salford
        "E08000007",  # Stockport
        "E08000008",  # Tameside
        "E08000009",  # Trafford
        "E08000010",
    ]  # Wigan
    areas = inner_london + outer_london + west_midlands + greater_manchester
    weight = np.array(
        [3.7] * len(inner_london + outer_london)
        + [1.0] * len(west_midlands + greater_manchester)
    )

    pop_size = N.loc[areas, age_group]
    seed = np.round(n_seed * pop_size / pop_size.sum())

    seeding = pd.Series(np.zeros_like(N), index=N.index)
    seeding.loc[areas, age_group] = seed  # Scatter
    return seeding.to_numpy()


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

    rv = pd.DataFrame(
        {
            "Group": "Lancaster",
            "Scenario": "Forecast",
            "CreationDate": creation_date,
            "DateOfValue": date,
            "Geography": "England",
            "ValueType": "CumCases",
            "Value": cases_mean,
            "LowerBound": cases_q[0],
            "UpperBound": cases_q[1],
        }
    )

    rv = rv[
        np.logical_and(
            rv["DateOfValue"] >= required_dates[0],
            rv["DateOfValue"] < required_dates[1],
        )
    ]

    return rv


def save_sims(dates, sims, la_names, age_groups, filename):
    f = h5py.File(filename, "w")
    dset_sim = f.create_dataset(
        "prediction", data=sims, compression="gzip", compression_opts=4
    )
    la_long = np.repeat(la_names, age_groups.shape[0]).astype(np.string_)
    age_long = np.tile(age_groups, la_names.shape[0]).astype(np.string_)
    dset_dims = f.create_dataset(
        "dimnames", data=[b"sim_id", b"t", b"la_age", b"state"]
    )
    dset_la = f.create_dataset("la_names", data=la_long)
    dset_age = f.create_dataset("age_names", data=age_long)
    dset_times = f.create_dataset("date", data=dates.astype(np.string_))
    f.close()


def extract_locs(in_file: str, out_file: str, loc: list):

    f = h5py.File(in_file, "r")
    la_names = f["la_names"][:].astype(str)
    la_loc = np.isin(la_names, loc)

    extract = f["prediction"][:, :, la_loc, :]

    save_sims(
        f["date"][:],
        extract,
        f["la_names"][la_loc],
        f["age_names"][la_loc],
        out_file,
    )
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
        "E08000015",  # Wirral
    ]
    extract_locs(in_file, out_file, la_seq)


@tf.function
def generate_case_numbers(n, rate):

    dtype = dtype_util.common_dtype([n, rate], dtype_hint=tf.float64)
    n = tf.convert_to_tensor(n, dtype=dtype)
    rate = tf.convert_to_tensor(rate, dtype=dtype)

    def cond(n_, i_, accum_):
        return tf.greater(tf.reduce_sum(n_), tf.constant(0.0, dtype=dtype))

    def body(n_, i_, accum_):
        new_n = tfd.Binomial(
            n_, probs=tf.constant(1.0, dtype=dtype) - tf.math.exp(-rate)
        ).sample()
        accum_ = accum_.write(i_, new_n)
        return n_ - new_n, i_ + 1, accum_

    accum = tf.TensorArray(dtype=n.dtype, size=20, dynamic_size=True)
    n, i, accum = tf.while_loop(cond, body, (n, 0, accum))
    return accum.gather(tf.range(i))


def initialise_previous_events_one_time(events, rate):
    past_events = generate_case_numbers(events.to_numpy(), rate)

    time_index = events.index.get_level_values(0)[0] - pd.to_timedelta(
        np.arange(past_events.shape[0]) + 1, "D"
    )
    new_index = pd.MultiIndex.from_product(
        [
            time_index,
            events.index.get_level_values(1).unique(),
            events.index.get_level_values(2).unique(),
        ]
    )
    past_events = pd.Series(
        past_events.numpy().flatten(), index=new_index, name="n"
    )
    print(".", flush=True, end="")
    return past_events


def initialise_previous_events(events, rate):
    """Samples imputed previous event times given a Markov rate.  All
    events are assumed independent.
    :param events: a pandas timeseries with index [date, space, age]
    :param rate: a Markov transition rate.
    """
    print("<generate_case_numbers>", flush=True, end="")
    events = events.groupby(level=0, axis=0).apply(
        lambda cases: initialise_previous_events_one_time(cases, 0.5)
    )
    events = events.sum(level=list(range(1, 4)))
    return events


def squared_jumping_distance(chain):
    diff = chain[1:] - chain[:-1]
    cumdiff = np.cumsum(diff, axis=-1)
    sqjumpdist = np.sum(cumdiff, axis=-1) ** 2
    return sqjumpdist


def p_null(results):
    accepted = results[:, 1] == 1.0
    pnull = np.mean(results[accepted, 2:].sum(axis=-1) == 0)
    return pnull


def jump_summary(posterior_file):
    f = h5py.File(posterior_file, "r")

    # SJD
    sjd_se = squared_jumping_distance(f["samples/events"][..., 0])
    sjd_ei = squared_jumping_distance(f["samples/events"][..., 1])

    # Acceptance
    accept_se = np.mean(f["acceptance/S->E"][:, 1])
    accept_ei = np.mean(f["acceptance/E->I"][:, 1])

    # Pr(null move | accepted)
    p_null_se = p_null(f["acceptance/S->E"])
    p_null_ei = p_null(f["acceptance/E->I"])

    f.close()
    return {
        "S->E": {
            "sjd": np.mean(sjd_se),
            "accept": accept_se,
            "p_null": p_null_se,
        },
        "E->I": {
            "sjd": np.mean(sjd_ei),
            "accept": accept_ei,
            "p_null": p_null_ei,
        },
    }


def plot_event_posterior(posterior, simulation, metapopulation=0):
    import matplotlib.pyplot as plt

    idx = np.linspace(0, posterior["samples/events"].shape[0] - 1, 200)
    fig, ax = plt.subplots(2, 2)

    ax[0][0].plot(
        posterior["samples/events"][idx, metapopulation, :, 0].T,
        color="lightblue",
        alpha=0.1,
    )
    ax[0][0].plot(
        simulation["events"][:, metapopulation, 0, 1],
        color="black",
        label="True events",
    )

    ax[0][1].plot(
        np.cumsum(
            posterior["samples/events"][idx, metapopulation, :, 0].T, axis=0
        ),
        color="lightblue",
        alpha=0.1,
    )
    ax[0][1].plot(
        np.cumsum(simulation["events"][:, metapopulation, 0, 1]),
        color="black",
        label="True events",
    )

    ax[1][0].plot(
        posterior["samples/events"][idx, metapopulation, :, 1].T,
        color="lightblue",
        alpha=0.1,
    )
    ax[1][0].plot(
        simulation["events"][:, metapopulation, 1, 2],
        color="black",
        label="True events",
    )

    ax[1][1].plot(
        np.cumsum(
            posterior["samples/events"][idx, metapopulation, :, 1].T, axis=0
        ),
        color="lightblue",
        alpha=0.1,
    )
    ax[1][1].plot(
        np.cumsum(simulation["events"][:, metapopulation, 1, 2]),
        color="black",
        label="True events",
    )

    ax[0][0].set_title("Events per day")
    ax[0][1].set_title("Cumulative events")

    ax[0][0].set_ylabel("S->E")
    ax[1][0].set_ylabel("E->I")

    return fig, ax


def distribute_geom(events, rate, delta_t=1.0):
    """Given a tensor `events`, returns a tensor of shape `events.shape + [t]`
    representing the events distributed over a number of days given geometric
    waiting times with rate `1-exp(-rate*delta_t)`"""

    events = tf.convert_to_tensor(events)
    rate = tf.convert_to_tensor(rate, dtype=events.dtype)

    accum = tf.TensorArray(events.dtype, size=0, dynamic_size=True)
    prob = 1.0 - tf.exp(-rate * delta_t)

    def body(i, events_, accum_):
        rv = tfd.Binomial(total_count=events_, probs=prob)
        failures = rv.sample()
        accum_ = accum_.write(i, failures)
        i += 1
        return i, events_ - failures, accum_

    def cond(_1, events_, _2):
        res = tf.reduce_sum(events_) > tf.constant(0, dtype=events.dtype)
        return res

    _1, _2, accum = tf.while_loop(cond, body, loop_vars=[1, events, accum])
    accum = accum.stack()

    return tf.transpose(accum, perm=(1, 0, 2))


def reduce_diagonals(m):
    def fn(m_):
        idx = (
            tf.range(m_.shape[-1])
            - tf.range(m_.shape[-2])[:, tf.newaxis]
            + m_.shape[-2]
            - 1
        )
        idx = tf.expand_dims(idx, axis=-1)
        return tf.scatter_nd(idx, m_, [m_.shape[-2] + m_.shape[-1] - 1])

    return tf.vectorized_map(fn, m)


def impute_previous_cases(events, rate, delta_t=1.0):
    """Imputes previous numbers of cases by using a geometric distribution

    :param events: a [M, T] tensor
    :param rate: the failure rate per `delta_t`
    :param delta_t: the size of the time step
    :returns: a tuple containing the matrix of events and the maximum
              number of timesteps into the past to allow padding of `events`.
    """
    prev_case_distn = distribute_geom(events, rate, delta_t)
    prev_cases = reduce_diagonals(prev_case_distn)

    # Trim preceding zero days
    total_events = tf.reduce_sum(prev_cases, axis=-2)
    num_zero_days = total_events.shape[-1] - tf.math.count_nonzero(
        tf.cumsum(total_events, axis=-1)
    )
    return (
        prev_cases[..., num_zero_days:],
        prev_case_distn.shape[-2] - num_zero_days,
    )


def mean_sojourn(in_events, out_events, init_state):
    """Calculated the mean sojourn time for individuals in a state
    within `in_events` and `out_events` given initial state `init_state`"""

    # state.shape = [..., M, T]
    state = (
        tf.cumsum(in_events - out_events, axis=-1, exclusive=True) + init_state
    )
    state = tf.reduce_sum(state, axis=(-2, -1))
    events = tf.reduce_sum(out_events, axis=(-2, -1))

    return 1.0 + state / events


def regularize_occults(events, occults, init_state, stoichiometry):
    """Regularizes an occult matrix such that counting
    processes are valid

    :param events: a [M, T, X] events tensor
    :param occults: a [M, T, X] occults tensor
    :param init_state: a [M, S] initial state tensor
    :param stoichiometry: a [X, S] stoichiometry matrix
    :returns: an tuple containing updated (state, occults) tensors
    """

    from covid.impl.util import compute_state

    def body(state_, occults_):
        state_t1 = tf.roll(state_, shift=-1, axis=-2)
        neg_state_idx = tf.where(state_t1 < 0)

        first_neg_state_idx = tf.gather(
            neg_state_idx,
            tf.concat(
                [
                    [[0]],
                    tf.where(neg_state_idx[:-1, 0] - neg_state_idx[1:, 0]) + 1,
                ],
                axis=0,
            ),
        )

        mask = tf.scatter_nd(
            first_neg_state_idx,
            tf.ones([first_neg_state_idx.shape[0], 1], dtype=state_t1.dtype),
            state_t1.shape,
        )
        delta_occults = tf.einsum("mts,xs->mtx", state_t1 * mask, stoichiometry)
        new_occults = tf.clip_by_value(
            occults_ - delta_occults, clip_value_min=0.0, clip_value_max=1.0e6
        )
        new_state = compute_state(
            init_state, events + new_occults, stoichiometry
        )
        return new_state, new_occults

    def cond(state_, _):
        return tf.reduce_any(state_ < 0)

    state = compute_state(init_state, events + occults, stoichiometry)
    new_state, new_occults = tf.while_loop(cond, body, (state, occults))

    return new_state, new_occults
