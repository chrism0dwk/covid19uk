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

    from gemlib.util import compute_state

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
