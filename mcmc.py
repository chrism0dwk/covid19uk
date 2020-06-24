"""MCMC Test Rig for COVID-19 UK model"""

import os
import pickle as pkl

import h5py
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tqdm
import yaml

from covid import config
from covid.model import load_data, CovidUKStochastic
from covid.util import sanitise_parameter, sanitise_settings
from covid.impl.util import make_transition_matrix
from covid.impl.mcmc import UncalibratedLogRandomWalk, random_walk_mvnorm_fn
from covid.impl.event_time_mh import EventTimesUpdate

tfd = tfp.distributions
tfb = tfp.bijectors

DTYPE = config.floatX

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tf.random.set_seed(2)

# Random moves of events.  What invalidates an epidemic, how can we test for it?
with open("ode_config.yaml", "r") as f:
    config = yaml.load(f)

param = sanitise_parameter(config["parameter"])
param = {k: tf.constant(v, dtype=DTYPE) for k, v in param.items()}

settings = sanitise_settings(config["settings"])

data = load_data(config["data"], settings, DTYPE)
data["pop"] = data["pop"].sum(level=0)

model = CovidUKStochastic(
    C=data["C"],
    N=data["pop"]["n"].to_numpy(),
    W=data["W"],
    date_range=settings["inference_period"],
    holidays=settings["holiday"],
    lockdown=settings["lockdown"],
    time_step=1.0,
)


# Load data
with open("stochastic_sim_covid.pkl", "rb") as f:
    example_sim = pkl.load(f)

event_tensor = example_sim["events"]  # shape [T, M, S, S]
num_times = event_tensor.shape[0]
num_meta = event_tensor.shape[1]
state_init = example_sim["state_init"]
se_events = event_tensor[:, :, 0, 1]
ei_events = event_tensor[:, :, 1, 2]
ir_events = event_tensor[:, :, 2, 3]


def logp(par, events):
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
    with tf.name_scope("main_log_p"):
        event_tensor = make_transition_matrix(
            events, [[0, 1], [1, 2], [2, 3]], [num_times, num_meta, 4]
        )
        y_logp = tf.reduce_sum(model.log_prob(event_tensor, p, state_init))
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
        return EventTimesUpdate(
            target_log_prob_fn=logp,
            target_event_id=target_event_id,
            prev_event_id=prev_event_id,
            next_event_id=next_event_id,
            dmax=2,
            mmax=2,
            nmax=10,
            initial_state=state_init,
        )

    return kernel_func


def is_accepted(result):
    if hasattr(result, "is_accepted"):
        return tf.cast(result.is_accepted, DTYPE)
    else:
        return is_accepted(result.inner_results)


def trace_results_fn(results):
    log_prob = results.proposed_results.target_log_prob
    accepted = is_accepted(results)
    q_ratio = results.proposed_results.log_acceptance_correction
    proposed = results.proposed_results.extra
    return tf.concat([[log_prob], [accepted], [q_ratio], proposed], axis=0)


@tf.function(autograph=False, experimental_compile=True)
def sample(n_samples, init_state, par_scale):
    init_state = init_state.copy()
    par_func = make_parameter_kernel(par_scale, 0.95)
    se_func = make_events_step(0, None, 1)
    ei_func = make_events_step(1, 0, 2)

    # Based on Gibbs idea posted by Pavel Sountsov
    # https://github.com/tensorflow/probability/issues/495
    results = ei_func(lambda s: logp(init_state[0], s)).bootstrap_results(init_state[1])

    samples_arr = [tf.TensorArray(s.dtype, size=n_samples) for s in init_state]
    results_arr = [tf.TensorArray(DTYPE, size=n_samples) for r in range(3)]

    def body(i, state, prev_results, samples, results):
        # Parameters
        def par_logp(par_state):
            state[0] = par_state  # close over state from outer scope
            return logp(*state)

        state[0], par_results = par_func(par_logp).one_step(state[0], prev_results)

        # States
        def state_logp(event_state):
            state[1] = event_state
            return logp(*state)

        state[1], se_results = se_func(state_logp).one_step(state[1], par_results)
        state[1], ei_results = ei_func(state_logp).one_step(state[1], se_results)

        samples = [samples[k].write(i, s) for k, s in enumerate(state)]
        results = [
            results[k].write(i, trace_results_fn(r))
            for k, r in enumerate([par_results, se_results, ei_results])
        ]
        return i + 1, state, ei_results, samples, results

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

if tf.test.gpu_device_name():
    print("Using GPU")
else:
    print("Using CPU")

NUM_LOOP_ITERATIONS = 500
NUM_LOOP_SAMPLES = 100
current_state = [
    np.array([0.6, 0.25], dtype=DTYPE),
    tf.stack([se_events, ei_events, ir_events], axis=-1),
]

posterior = h5py.File(os.path.expandvars(config["output"]["posterior"]), "w")
event_size = [NUM_LOOP_ITERATIONS * NUM_LOOP_SAMPLES] + list(current_state[1].shape)
par_samples = posterior.create_dataset(
    "samples/parameter",
    [NUM_LOOP_ITERATIONS * NUM_LOOP_SAMPLES, current_state[0].shape[0]],
    dtype=np.float64,
)
se_samples = posterior.create_dataset("samples/events", event_size, dtype=DTYPE)
par_results = posterior.create_dataset(
    "acceptance/parameter", (NUM_LOOP_ITERATIONS * NUM_LOOP_SAMPLES, 152), dtype=DTYPE,
)
se_results = posterior.create_dataset(
    "acceptance/S->E", (NUM_LOOP_ITERATIONS * NUM_LOOP_SAMPLES, 152), dtype=DTYPE
)
ei_results = posterior.create_dataset(
    "acceptance/E->I", (NUM_LOOP_ITERATIONS * NUM_LOOP_SAMPLES, 152), dtype=DTYPE
)

print("Initial logpi:", logp(*current_state))
par_scale = tf.linalg.diag(
    tf.ones(current_state[0].shape, dtype=current_state[0].dtype) * 0.1
)

# We loop over successive calls to sample because we have to dump results
#   to disc, or else end OOM (even on a 32GB system).
for i in tqdm.tqdm(range(NUM_LOOP_ITERATIONS), unit_scale=NUM_LOOP_SAMPLES):
    # with tf.profiler.experimental.Profile("/tmp/tf_logdir"):
    samples, results = sample(
        NUM_LOOP_SAMPLES, init_state=current_state, par_scale=par_scale
    )
    current_state = [s[-1] for s in samples]
    s = slice(i * NUM_LOOP_SAMPLES, i * NUM_LOOP_SAMPLES + NUM_LOOP_SAMPLES)
    par_samples[s, ...] = samples[0].numpy()
    cov = np.cov(
        np.log(par_samples[: (i * NUM_LOOP_SAMPLES + NUM_LOOP_SAMPLES), ...]),
        rowvar=False,
    )
    print(current_state[0].numpy())
    print(cov)
    if np.all(np.isfinite(cov)):
        par_scale = 2.38 ** 2 * cov / 2.0

    se_samples[s, ...] = samples[1].numpy()
    par_results[s, ...] = results[0].numpy()
    se_results[s, ...] = results[1].numpy()
    ei_results[s, ...] = results[2].numpy()

    print("Acceptance0:", tf.reduce_mean(tf.cast(results[0][:, 1], tf.float32)))
    print("Acceptance1:", tf.reduce_mean(tf.cast(results[1][:, 1], tf.float32)))
    print("Acceptance2:", tf.reduce_mean(tf.cast(results[2][:, 1], tf.float32)))

print(f"Acceptance param: {par_results[:, 1].mean()}")
print(f"Acceptance S->E: {se_results[:, 1].mean()}")
print(f"Acceptance E->I: {ei_results[:, 1].mean()}")

posterior.close()
