import tqdm
import pandas as pd
import matplotlib.pyplot as plt
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
from covid.impl.util import make_transition_rate_matrix
from covid.impl.mcmc import UncalibratedEventTimesUpdate
from covid.impl.discrete_markov import events_to_full_transitions
from covid.pydata import phe_linelist_timeseries, zero_cases, collapse_pop

DTYPE = config.floatX

phe = phe_linelist_timeseries('/home/jewellcp/Insync/jewellcp@lancaster.ac.uk/OneDrive Biz - Shared/covid19/data/PHE_2020-04-08/Anonymised Line List 20200409.csv')
pop = collapse_pop('data/c2019modagepop.csv')
cases = zero_cases(phe, pop)

#from covid.util import initialise_previous_events
#print("Initialising ei...", flush=True, end='')
#ei = initialise_previous_events(cases, 0.5)
#ei = zero_cases(ei, pop)
#print("Done")
#print("Initialising ir...", flush=True, end='')
#se = initialise_previous_events(ei, 0.25)
#se = zero_cases(se, pop)
#print("Done")

ei = cases.copy()
se = cases.copy()
idx_ei = cases.index.to_frame().copy()
idx_ei['date'] = idx_ei['date'] - pd.to_timedelta(2, 'D')
ei.index = pd.MultiIndex.from_frame(idx_ei)

idx_se = cases.index.to_frame().copy()
idx_se['date'] = idx_se['date'] - pd.to_timedelta(6, 'D')
se.index = pd.MultiIndex.from_frame(idx_se)


events = pd.concat([se, ei, cases], axis=1)
events.columns = ['se','ei','ir']
events.index.set_names(['date','UTLA_code','Age'], range(3), inplace=True)
# # events.to_csv('tmp_events.csv')
#
# events = pd.read_csv('tmp_events.csv')
# events.index = pd.MultiIndex.from_frame(events[['date','UTLA_code','Age']])
# events = events[['se','ei','ir']]
events[events.isna()] = 0.0
num_times = events.index.get_level_values(0).unique().shape[0]

se_events = events['se'].to_numpy().reshape([num_times,2533])
ei_events = events['ei'].to_numpy().reshape([num_times,2533])
ir_events = events['ir'].to_numpy().reshape([num_times,2533])

event_tensor = make_transition_rate_matrix([se_events, ei_events, ir_events], [[0,1],[1,2],[2,3]],
                                           tf.zeros([num_times, 2533,4]))
event_tensor = tf.cast(event_tensor, dtype=DTYPE)
# 53, 1342

# Fill in susceptibles
print("Event_tensor: ", event_tensor)
print(f"Sum cases: {events.to_numpy().sum()}")
print(f"Sum events: {tf.reduce_sum(event_tensor)}")

init_state = tf.zeros([4])
timeline = tf.cumsum(tf.reduce_sum(event_tensor, axis=[1, -2]), axis=0)

# Random moves of events.  What invalidates an epidemic, how can we test for it?
with open('ode_config.yaml','r') as f:
    config = yaml.load(f)

param = sanitise_parameter(config['parameter'])
param = {k: tf.constant(v, dtype=DTYPE) for k, v in param.items()}

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

seeding = seed_areas(data['pop']['n'])  # Seed 40-44 age group, 30 seeds by popn size
state_init = model.create_initial_state(init_matrix=seeding)

def logp(event_tensor):
    p = param
    #p['beta1'] = tf.convert_to_tensor(par[0], dtype=DTYPE)
    #p['gamma'] = tf.convert_to_tensor(par[1], dtype=DTYPE)
    beta_logp = tfd.Gamma(concentration=tf.constant(1., dtype=DTYPE),
                          rate=tf.constant(1., dtype=DTYPE)).log_prob(p['beta1'])
    gamma_logp = tfd.Gamma(concentration=tf.constant(100., dtype=DTYPE),
                           rate=tf.constant(400., dtype=DTYPE)).log_prob(p['gamma'])
    y_logp = tf.reduce_sum(model.log_prob(event_tensor, p, state_init))
    logp = beta_logp + gamma_logp + y_logp
    return logp


def random_walk_mvnorm_fn(covariance, p_u=0.95, name=None):
    """Returns callable that adds Multivariate Normal noise to the input
    :param covariance: the covariance matrix of the mvnorm proposal
    :param p_u: the bounded convergence parameter.  If equal to 1, then all proposals are drawn from the
                MVN(0, covariance) distribution, if less than 1, proposals are drawn from MVN(0, covariance)
                with probabilit p_u, and MVN(0, 0.1^2I_d/d) otherwise.
    :returns: a function implementing the proposal.
    """
    covariance = covariance + tf.eye(covariance.shape[0], dtype=DTYPE) * 1.e-9
    scale_tril = tf.linalg.cholesky(covariance)
    rv_adapt = tfp.distributions.MultivariateNormalTriL(loc=tf.zeros(covariance.shape[0], dtype=DTYPE),
                                                        scale_tril=scale_tril)
    rv_fix = tfp.distributions.Normal(loc=tf.zeros(covariance.shape[0], dtype=DTYPE),
                                      scale=0.01/covariance.shape[0],)
    u = tfp.distributions.Bernoulli(probs=p_u)

    def _fn(state_parts, seed):
        with tf.name_scope(name or 'random_walk_mvnorm_fn'):
            def proposal():
                rv = tf.stack([rv_fix.sample(), rv_adapt.sample()])
                uv = u.sample()
                return tf.gather(rv, uv)
            new_state_parts = [proposal() + state_part for state_part in state_parts]
            return new_state_parts

    return _fn

unconstraining_bijector = [tfb.Exp()]
initial_mcmc_state = event_tensor  # tf.constant([0.09, 0.5], dtype=DTYPE)  # beta1, gamma, I0
print("Initial log likelihood:", logp(event_tensor))

def trace_fn(state, prev_results):
    return (prev_results.is_accepted,
            prev_results.accepted_results.target_log_prob)

def parameter_kernel(scale, bounded_convergence):
    return tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=tfp.mcmc.RandomWalkMetropolis(
                target_log_prob_fn=logp,
                new_state_fn=random_walk_mvnorm_fn(scale, p_u=bounded_convergence)
            ),
        bijector=unconstraining_bijector)


def events_kernel(q, p, alpha, coord):
    return tfp.mcmc.MetropolisHastings(
        inner_kernel=UncalibratedEventTimesUpdate(target_log_prob_fn=logp,
                                                  transition_coord=coord,
                                                  q=q,
                                                  p=p,
                                                  alpha=alpha)
    )

@tf.function  # (autograph=False, experimental_compile=True)
def sample(n_samples, init_state):
    kernel1 = events_kernel(q=500./192508., p=0.2, alpha=0.5, coord=[0, 1])
    kernel2 = events_kernel(q=500./192508., p=0.2, alpha=0.5, coord=[1, 2])
    results = kernel1.bootstrap_results(init_state)

    samples_arr = [tf.TensorArray(init_state.dtype, size=n_samples, colocate_with_first_write_call=False),
               tf.TensorArray(init_state.dtype, size=n_samples, colocate_with_first_write_call=False)]
    results_arr = [tf.TensorArray(tf.bool, size=n_samples, colocate_with_first_write_call=False),
               tf.TensorArray(tf.bool, size=n_samples, colocate_with_first_write_call=False)]

    def body(i, state, prev_results, samples, results):
        new_state, new_results1 = kernel1.one_step(state, prev_results)
        new_state, new_results2 = kernel2.one_step(new_state, new_results1)
        samples[0] = samples[0].write(i, new_state[..., 0, 1])  # S->E
        samples[1] = samples[1].write(i, new_state[..., 1, 2])  # E->I
        results[0] = results[0].write(i, new_results1.is_accepted)
        results[1] = results[1].write(i, new_results2.is_accepted)
        return i+1, new_state, new_results2, samples, results

    def cond(i, _1, _2, _3, _4):
        return i < n_samples

    _1, _2, _3, samples, results = tf.while_loop(cond=cond, body=body,
                                                 loop_vars=[0, init_state, results, samples_arr, results_arr])

    return tf.stack([s.stack() for s in samples]), tf.stack([r.stack() for r in results])


if __name__=='__main__':

    num_loop_iterations = 100
    num_loop_samples = 100

    posterior = h5py.File('posterior.h5','w')
    event_size = [num_loop_iterations * num_loop_samples] + event_tensor[:, :, 0, 1].shape
    se_samples = posterior.create_dataset('samples/S->E', event_size.as_list(), dtype=DTYPE)
    ei_samples = posterior.create_dataset('samples/E->I', event_size.as_list(), dtype=DTYPE)
    se_results = posterior.create_dataset('acceptance/S->E', (num_loop_iterations * num_loop_samples,), dtype=np.bool)
    ei_results = posterior.create_dataset('acceptance/E->I', (num_loop_iterations * num_loop_samples,), dtype=np.bool)

    current_state = event_tensor
    for i in tqdm.tqdm(range(num_loop_iterations), unit_scale=100):
        samples, results = sample(num_loop_samples, init_state=current_state)
        s = slice(i*num_loop_samples, i*num_loop_samples+num_loop_samples)
        se_samples[s,...] = samples[0].numpy()
        ei_samples[s,...] = samples[1].numpy()
        se_results[s] = results[0].numpy()
        ei_results[s] = results[1].numpy()
        print("Acceptance0:", tf.reduce_mean(tf.cast(results[0], tf.float32)))
        print("Acceptance1:", tf.reduce_mean(tf.cast(results[1], tf.float32)))

    print(f'Acceptance0 total: {se_results[:].mean()}')
    print(f'Acceptance1 total: {ei_results[:].mean()}')

    posterior.close()

