import os
import tqdm
import pickle as pkl
import yaml
import h5py

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from covid import config
from covid.model import load_data, CovidUKStochastic
from covid.util import sanitise_parameter, sanitise_settings, seed_areas
from covid.impl.util import make_transition_matrix
from covid.impl.mcmc import UncalibratedEventTimesUpdate, UncalibratedLogRandomWalk, MH_within_Gibbs, \
    random_walk_mvnorm_fn

DTYPE = config.floatX


# Random moves of events.  What invalidates an epidemic, how can we test for it?
with open('ode_config.yaml','r') as f:
    config = yaml.load(f)

param = sanitise_parameter(config['parameter'])
param = {k: tf.constant(v, dtype=DTYPE) for k, v in param.items()}

settings = sanitise_settings(config['settings'])

data = load_data(config['data'], settings, DTYPE)
data['pop'] = data['pop'].sum(level=0)

model = CovidUKStochastic(C=data['C'],
                          N=data['pop']['n'].to_numpy(),
                          W=data['W'],
                          date_range=settings['inference_period'],
                          holidays=settings['holiday'],
                          lockdown=settings['lockdown'],
                          time_step=1.)

# Load data
with open('stochastic_sim.pkl','rb') as f:
    example_sim = pkl.load(f)

event_tensor = example_sim['events']  # shape [T, M, S, S]
num_times = event_tensor.shape[0]
num_meta = event_tensor.shape[1]
state_init = example_sim['state_init']
se_events = event_tensor[:, :, 0, 1]
ei_events = event_tensor[:, :, 1, 2]
ir_events = event_tensor[:, :, 2, 3]

def logp(se, ei):
    p = param
    #p['beta1'] = tf.convert_to_tensor(par[0], dtype=DTYPE)
    #p['beta2'] = tf.convert_to_tensor(par[1], dtype=DTYPE)
    #p['beta3'] = tf.convert_to_tensor(par[2], dtype=DTYPE)
    #p['gamma'] = tf.convert_to_tensor(par[3], dtype=DTYPE)
    #beta1_logp = tfd.Gamma(concentration=tf.constant(1., dtype=DTYPE),
    #                      rate=tf.constant(1., dtype=DTYPE)).log_prob(p['beta1'])
    #beta2_logp = tfd.Gamma(concentration=tf.constant(1., dtype=DTYPE),
    #                       rate=tf.constant(1., dtype=DTYPE)).log_prob(p['beta2'])
    # beta3_logp = tfd.Gamma(concentration=tf.constant(2., dtype=DTYPE),
    #                       rate=tf.constant(2., dtype=DTYPE)).log_prob(p['beta3'])
    # gamma_logp = tfd.Gamma(concentration=tf.constant(100., dtype=DTYPE),
    #                       rate=tf.constant(400., dtype=DTYPE)).log_prob(p['gamma'])
    event_tensor = make_transition_matrix([se, ei, ir_events],  # ir_events global scope
                                          [[0, 1], [1, 2], [2, 3]],
                                          tf.zeros([num_times, num_meta, 4]))  # Todo: remove constant
    y_logp = tf.reduce_sum(model.log_prob(event_tensor, p, state_init))
    logp = y_logp
    return logp

print("Initial logpi:", logp(se_events, ei_events))


def trace_fn(state, prev_results):
    return (prev_results.is_accepted,
            prev_results.accepted_results.target_log_prob)


# Pavel's suggestion for a Gibbs kernel requires
# kernel factory functions.
def make_parameter_kernel(scale, bounded_convergence):
    def kernel_func(logp):
        return tfp.mcmc.MetropolisHastings(
            inner_kernel=UncalibratedLogRandomWalk(
                    target_log_prob_fn=logp,
                    new_state_fn=random_walk_mvnorm_fn(scale, p_u=bounded_convergence)
                ), name='parameter_update')
    return kernel_func


def make_events_step(p, alpha):
    def kernel_func(logp):
        return tfp.mcmc.MetropolisHastings(
            inner_kernel=UncalibratedEventTimesUpdate(target_log_prob_fn=logp,
                                                      p=p,
                                                      alpha=alpha)
        )
    return kernel_func


def is_accepted(result):
    if hasattr(result, 'is_accepted'):
        return result.is_accepted
    else:
        return is_accepted(result.inner_results)


@tf.function(autograph=False, experimental_compile=True)
def sample(n_samples, init_state, par_scale):
    init_state = init_state.copy()
    par_func = make_parameter_kernel(par_scale, 0.95)
    kernel_func1 = make_events_step(p=0.0001, alpha=0.9)
    kernel_func2 = make_events_step(p=0.0001, alpha=0.9)

    # Based on Gibbs idea posted by Pavel Sountsov https://github.com/tensorflow/probability/issues/495
    gibbs = MH_within_Gibbs(logp, [kernel_func1, kernel_func2])
    results = gibbs.bootstrap_results(init_state)

    samples_arr = [tf.TensorArray(s.dtype, size=n_samples) for s in init_state]
    results_arr = [tf.TensorArray(tf.bool, size=n_samples) for r in results]

    def body(i, state, prev_results, samples, results):
        new_state, new_results = gibbs.one_step(state, prev_results)
        samples = [samples[k].write(i, s) for k, s in enumerate(new_state)]
        results = [results[k].write(i, is_accepted(r)) for k, r in enumerate(new_results)]
        return i+1, new_state, new_results, samples, results

    def cond(i, _1, _2, _3, _4):
        return i < n_samples

    _1, _2, _3, samples, results = tf.while_loop(cond=cond, body=body,
                                                 loop_vars=[0, init_state, results, samples_arr, results_arr])

    return [s.stack() for s in samples], [r.stack() for r in results]


if __name__=='__main__':





    num_loop_iterations = 50
    num_loop_samples = 100
    current_state = [se_events, ei_events]

    posterior = h5py.File('posterior.h5','w')
    event_size = [num_loop_iterations * num_loop_samples] + list(current_state[1].shape)
    par_samples = posterior.create_dataset('samples/parameter', [num_loop_iterations*num_loop_samples,
                                                                 current_state[0].shape[0]], dtype=np.float64)
    se_samples = posterior.create_dataset('samples/S->E', event_size, dtype=DTYPE)
    ei_samples = posterior.create_dataset('samples/E->I', event_size, dtype=DTYPE)
    par_results = posterior.create_dataset('acceptance/parameter', (num_loop_iterations * num_loop_samples,), dtype=np.bool)
    se_results = posterior.create_dataset('acceptance/S->E', (num_loop_iterations * num_loop_samples,), dtype=np.bool)
    ei_results = posterior.create_dataset('acceptance/E->I', (num_loop_iterations * num_loop_samples,), dtype=np.bool)

    par_scale = tf.linalg.diag(tf.ones(current_state[0].shape, dtype=current_state[0].dtype) * 0.1)

    # We loop over successive calls to sample because we have to dump results
    #   to disc, or else end OOM (even on a 32GB system).
    for i in tqdm.tqdm(range(num_loop_iterations), unit_scale=num_loop_samples):
        samples, results = sample(num_loop_samples, init_state=current_state, par_scale=par_scale)
        current_state = [s[-1] for s in samples]
        s = slice(i*num_loop_samples, i*num_loop_samples+num_loop_samples)
        #par_samples[s, ...] = samples[0].numpy()
        cov = np.cov(np.log(par_samples[:(i*num_loop_samples+num_loop_samples), ...]), rowvar=False)
        print(current_state[0].numpy())
        print(cov)
        # if(np.all(np.isfinite(cov))):
        #     par_scale = 2.38**2 * cov / 2.
        se_samples[s, ...] = samples[0].numpy()
        ei_samples[s, ...] = samples[1].numpy()
        #par_results[s, ...] = results[0].numpy()
        se_results[s, ...] = results[0].numpy()
        ei_results[s, ...] = results[1].numpy()

        print("Acceptance0:", tf.reduce_mean(tf.cast(results[0], tf.float32)))
        print("Acceptance1:", tf.reduce_mean(tf.cast(results[1], tf.float32)))
        #print("Acceptance2:", tf.reduce_mean(tf.cast(results[2], tf.float32)))

    #print(f'Acceptance param: {par_results[:].mean()}')
    print(f'Acceptance S->E: {se_results[:].mean()}')
    print(f'Acceptance E->I: {ei_results[:].mean()}')

    posterior.close()

