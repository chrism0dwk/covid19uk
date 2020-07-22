"""MCMC Test Rig for COVID-19 UK model"""
import optparse
import os
import pickle as pkl
from collections import OrderedDict
from time import perf_counter

import h5py
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tqdm
import yaml

from covid import config
from covid.model import load_data, CovidUKStochastic
from covid.pydata import phe_case_data
from covid.util import sanitise_parameter, sanitise_settings, impute_previous_cases
from covid.impl.mcmc import UncalibratedLogRandomWalk, random_walk_mvnorm_fn
from covid.impl.event_time_mh import UncalibratedEventTimesUpdate
from covid.impl.occult_events_mh import UncalibratedOccultUpdate

###########
# TF Bits #
###########

tfd = tfp.distributions
tfb = tfp.bijectors

DTYPE = config.floatX

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["XLA_FLAGS"] = '--xla_dump_to=xla_dump --xla_dump_hlo_pass_re=".*"'

if tf.test.gpu_device_name():
    print("Using GPU")
else:
    print("Using CPU")

# Read in settings
# parser = optparse.OptionParser()
# parser.add_option(
#     "--config",
#     "-c",
#     dest="config",
#     default="ode_config.yaml",
#     help="configuration file",
# )
# options, cmd_args = parser.parse_args()
# print("Loading config file:", options.config)

# with open(options.config, "r") as f:
with open("ode_config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

settings = sanitise_settings(config["settings"])

param = sanitise_parameter(config["parameter"])
param = {k: tf.constant(v, dtype=DTYPE) for k, v in param.items()}

covar_data = load_data(config["data"], settings, DTYPE)

cases = phe_case_data(config["data"]["reported_cases"], settings["inference_period"])
ei_events, lag_ei = impute_previous_cases(cases, 0.25)
se_events, lag_se = impute_previous_cases(ei_events, 0.25)
ir_events = np.pad(cases, ((0, 0), (lag_ei + lag_se - 2, 0)))
ei_events = np.pad(ei_events, ((0, 0), (lag_se - 1, 0)))


model = CovidUKStochastic(
    C=covar_data["C"],
    N=covar_data["pop"],
    W=covar_data["W"],
    date_range=settings["inference_period"],
    holidays=settings["holiday"],
    lockdown=settings["lockdown"],
    time_step=1.0,
)

##########################
# Log p and MCMC kernels #
##########################


def logp(par, events, occult_events):
    p = param
    p["beta1"] = tf.convert_to_tensor(par[0], dtype=DTYPE)
    # p['beta2'] = tf.convert_to_tensor(par[1], dtype=DTYPE)
    # p['beta3'] = tf.convert_to_tensor(par[2], dtype=DTYPE)
    p["gamma"] = tf.convert_to_tensor(par[1], dtype=DTYPE)
    beta1_logp = tfd.Gamma(
        concentration=tf.constant(1.0, dtype=DTYPE), rate=tf.constant(1.0, dtype=DTYPE)
    ).log_prob(p["beta1"])
    # beta2_logp = tfd.Gamma(concentration=tf.constant(1., dtype=DTYPE),
    #                       rate=tf.constant(1., dtype=DTYPE)).log_prob(p['beta2'])
    # beta3_logp = tfd.Gamma(concentration=tf.constant(2., dtype=DTYPE),
    #                       rate=tf.constant(2., dtype=DTYPE)).log_prob(p['beta3'])
    gamma_logp = tfd.Gamma(
        concentration=tf.constant(100.0, dtype=DTYPE),
        rate=tf.constant(400.0, dtype=DTYPE),
    ).log_prob(p["gamma"])
    with tf.name_scope("epidemic_log_posterior"):
        y_logp = model.log_prob(events + occult_events, p, state_init)
    logp = beta1_logp + gamma_logp + y_logp
    return logp


# Pavel's suggestion for a Gibbs kernel requires
# kernel factory functions.
def make_parameter_kernel(scale, bounded_convergence):
    def kernel_func(logp):
        return tfp.mcmc.MetropolisHastings(
            inner_kernel=UncalibratedLogRandomWalk(
                target_log_prob_fn=logp,
                new_state_fn=random_walk_mvnorm_fn(scale, p_u=bounded_convergence),
            ),
            name="parameter_update",
        )

    return kernel_func


def make_events_step(target_event_id, prev_event_id=None, next_event_id=None):
    def kernel_func(logp):
        return tfp.mcmc.MetropolisHastings(
            inner_kernel=UncalibratedEventTimesUpdate(
                target_log_prob_fn=logp,
                target_event_id=target_event_id,
                prev_event_id=prev_event_id,
                next_event_id=next_event_id,
                initial_state=state_init,
                dmax=config["mcmc"]["dmax"],
                mmax=config["mcmc"]["m"],
                nmax=config["mcmc"]["nmax"],
            ),
            name="event_update",
        )

    return kernel_func


def make_occults_step(target_event_id):
    def kernel_func(logp):
        return tfp.mcmc.MetropolisHastings(
            inner_kernel=UncalibratedOccultUpdate(
                target_log_prob_fn=logp,
                target_event_id=target_event_id,
                nmax=config["mcmc"]["occult_nmax"],
                t_range=[se_events.shape[1] - 21, se_events.shape[1]],
            ),
            name="occult_update",
        )

    return kernel_func


def is_accepted(result):
    if hasattr(result, "is_accepted"):
        return tf.cast(result.is_accepted, DTYPE)
    return is_accepted(result.inner_results)


def trace_results_fn(results):
    log_prob = results.proposed_results.target_log_prob
    accepted = is_accepted(results)
    q_ratio = results.proposed_results.log_acceptance_correction
    if hasattr(results.proposed_results, "extra"):
        proposed = tf.cast(results.proposed_results.extra, log_prob.dtype)
        return tf.concat([[log_prob], [accepted], [q_ratio], proposed], axis=0)
    return tf.concat([[log_prob], [accepted], [q_ratio]], axis=0)


def forward_results(prev_results, next_results):
    accepted_results = next_results.accepted_results._replace(
        target_log_prob=prev_results.accepted_results.target_log_prob
    )
    return next_results._replace(accepted_results=accepted_results)


@tf.function(autograph=False, experimental_compile=True)
def sample(n_samples, init_state, par_scale, num_event_updates):
    with tf.name_scope("main_mcmc_sample_loop"):
        init_state = init_state.copy()
        par_func = make_parameter_kernel(par_scale, 0.0)
        se_func = make_events_step(0, None, 1)
        ei_func = make_events_step(1, 0, 2)
        se_occult = make_occults_step(0)
        ei_occult = make_occults_step(1)

        # Based on Gibbs idea posted by Pavel Sountsov
        # https://github.com/tensorflow/probability/issues/495
        results = [
            par_func(lambda p: logp(p, init_state[1], init_state[2])).bootstrap_results(
                init_state[0]
            ),
            se_func(lambda s: logp(init_state[0], s, init_state[2])).bootstrap_results(
                init_state[1]
            ),
            ei_func(lambda s: logp(init_state[0], s, init_state[2])).bootstrap_results(
                init_state[1]
            ),
            se_occult(
                lambda s: logp(init_state[0], init_state[1], s)
            ).bootstrap_results(init_state[2]),
            ei_occult(
                lambda s: logp(init_state[0], init_state[1], s)
            ).bootstrap_results(init_state[2]),
        ]

        samples_arr = [tf.TensorArray(s.dtype, size=n_samples) for s in init_state]
        results_arr = [tf.TensorArray(DTYPE, size=n_samples) for r in range(5)]

        def body(i, state, results, sample_accum, results_accum):
            # Parameters
            def par_logp(par_state):
                state[0] = par_state  # close over state from outer scope
                return logp(*state)

            par_kernel = par_func(par_logp)
            state[0], results[0] = par_kernel.one_step(
                state[0], par_kernel.bootstrap_results(state[0])
            )

            # States
            results[4] = forward_results(results[0], results[4])

            def infec_body(j, state, results):
                def state_logp(event_state):
                    state[1] = event_state
                    return logp(*state)

                def occult_logp(occult_state):
                    state[2] = occult_state
                    return logp(*state)

                state[1], results[1] = se_func(state_logp).one_step(
                    state[1], forward_results(results[4], results[1])
                )
                state[1], results[2] = ei_func(state_logp).one_step(
                    state[1], forward_results(results[1], results[2])
                )
                state[2], results[3] = se_occult(occult_logp).one_step(
                    state[2], forward_results(results[2], results[3])
                )
                # results[3] = forward_results(results[2], results[3])
                state[2], results[4] = ei_occult(occult_logp).one_step(
                    state[2], forward_results(results[3], results[4])
                )
                # results[4] = forward_results(results[3], results[4])
                j += 1
                return j, state, results

            def infec_cond(j, state, results):
                return j < num_event_updates

            _, state, results = tf.while_loop(
                infec_cond,
                infec_body,
                loop_vars=[tf.constant(0, tf.int32), state, results],
            )

            sample_accum = [sample_accum[k].write(i, s) for k, s in enumerate(state)]
            results_accum = [
                results_accum[k].write(i, trace_results_fn(r))
                for k, r in enumerate(results)
            ]
            return i + 1, state, results, sample_accum, results_accum

        def cond(i, _1, _2, _3, _4):
            return i < n_samples

        _1, _2, _3, samples, results = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=[0, init_state, results, samples_arr, results_arr],
        )

        return [s.stack() for s in samples], [r.stack() for r in results]


##################
# MCMC loop here #
##################

# MCMC Control
NUM_BURSTS = config["mcmc"]["num_bursts"]
NUM_BURST_SAMPLES = config["mcmc"]["num_burst_samples"]
NUM_EVENT_TIME_UPDATES = config["mcmc"]["num_event_time_updates"]

# RNG stuff
tf.random.set_seed(2)

# Initial state.  NB [M, T, X] layout for events.
events = tf.stack([se_events, ei_events, ir_events], axis=-1)
state_init = tf.concat([model.N[:, tf.newaxis], events[:, 0, :]], axis=-1)
events = events[:, 1:, :]
current_state = [
    np.array([0.85, 0.25], dtype=DTYPE),
    events,
    tf.zeros_like(events),
]

# Output Files
posterior = h5py.File(
    os.path.expandvars(config["output"]["posterior"]),
    "w",
    rdcc_nbytes=1024 ** 2 * 400,
    rdcc_nslots=100000,
)
event_size = [NUM_BURSTS * NUM_BURST_SAMPLES] + list(current_state[1].shape)
# event_chunk = (10, 1, 1, 1)
# print("Event chunk size:", event_chunk)
par_samples = posterior.create_dataset(
    "samples/parameter",
    [NUM_BURSTS * NUM_BURST_SAMPLES, current_state[0].shape[0]],
    dtype=np.float64,
)
event_samples = posterior.create_dataset(
    "samples/events",
    event_size,
    dtype=DTYPE,
    chunks=(1024, 64, 64, current_state[1].shape[-1]),
    compression="lzf",
)
occult_samples = posterior.create_dataset(
    "samples/occults",
    event_size,
    dtype=DTYPE,
    chunks=(1024, 64, 64, current_state[1].shape[-1]),
    compression="lzf",
)

output_results = [
    posterior.create_dataset(
        "results/parameter", (NUM_BURSTS * NUM_BURST_SAMPLES, 3), dtype=DTYPE,
    ),
    posterior.create_dataset(
        "results/move/S->E",
        (NUM_BURSTS * NUM_BURST_SAMPLES, 3 + model.N.shape[0]),
        dtype=DTYPE,
    ),
    posterior.create_dataset(
        "results/move/E->I",
        (NUM_BURSTS * NUM_BURST_SAMPLES, 3 + model.N.shape[0]),
        dtype=DTYPE,
    ),
    posterior.create_dataset(
        "results/occult/S->E", (NUM_BURSTS * NUM_BURST_SAMPLES, 6), dtype=DTYPE
    ),
    posterior.create_dataset(
        "results/occult/E->I", (NUM_BURSTS * NUM_BURST_SAMPLES, 6), dtype=DTYPE
    ),
]

print("Initial logpi:", logp(*current_state))
par_scale = tf.linalg.diag(
    tf.ones(current_state[0].shape, dtype=current_state[0].dtype) * 0.1
)

# We loop over successive calls to sample because we have to dump results
#   to disc, or else end OOM (even on a 32GB system).
# with tf.profiler.experimental.Profile("/tmp/tf_logdir"):
for i in tqdm.tqdm(range(NUM_BURSTS), unit_scale=NUM_BURST_SAMPLES):
    samples, results = sample(
        NUM_BURST_SAMPLES,
        init_state=current_state,
        par_scale=par_scale,
        num_event_updates=tf.constant(NUM_EVENT_TIME_UPDATES, tf.int32),
    )
    current_state = [s[-1] for s in samples]
    s = slice(i * NUM_BURST_SAMPLES, i * NUM_BURST_SAMPLES + NUM_BURST_SAMPLES)
    par_samples[s, ...] = samples[0].numpy()
    cov = np.cov(
        np.log(par_samples[: (i * NUM_BURST_SAMPLES + NUM_BURST_SAMPLES), ...]),
        rowvar=False,
    )
    print(current_state[0].numpy())

    print(cov)
    if (i * NUM_BURST_SAMPLES) > 1000 and np.all(np.isfinite(cov)):
        par_scale = 2.38 ** 2 * cov / 2.0

    start = perf_counter()
    event_samples[s, ...] = samples[1].numpy()
    occult_samples[s, ...] = samples[2].numpy()
    end = perf_counter()

    for i, ro in enumerate(output_results):
        ro[s, ...] = results[i]

    print("Storage time:", end - start, "seconds")
    print("Acceptance par:", tf.reduce_mean(tf.cast(results[0][:, 1], tf.float32)))
    print(
        "Acceptance move S->E:", tf.reduce_mean(tf.cast(results[1][:, 1], tf.float32))
    )
    print(
        "Acceptance move E->I:", tf.reduce_mean(tf.cast(results[2][:, 1], tf.float32))
    )
    print(
        "Acceptance occult S->E:", tf.reduce_mean(tf.cast(results[3][:, 1], tf.float32))
    )
    print(
        "Acceptance occult E->I:", tf.reduce_mean(tf.cast(results[4][:, 1], tf.float32))
    )

print(f"Acceptance param: {output_results[0][:, 1].mean()}")
print(f"Acceptance move S->E: {output_results[1][:, 1].mean()}")
print(f"Acceptance move E->I: {output_results[2][:, 1].mean()}")
print(f"Acceptance occult S->E: {output_results[3][:, 1].mean()}")
print(f"Acceptance occult E->I: {output_results[4][:, 1].mean()}")

posterior.close()
