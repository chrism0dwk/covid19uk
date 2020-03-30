import optparse
import yaml
import time
import pickle as pkl
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
import matplotlib.pyplot as plt


from covid.rdata import load_population, load_age_mixing, load_mobility_matrix
from covid.pydata import load_commute_volume
from covid.model import CovidUKODE, covid19uk_logp
from covid.util import *

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



if __name__ == '__main__':

    parser = optparse.OptionParser()
    parser.add_option("--config", "-c", dest="config", default="ode_config.yaml",
                      help="configuration file")
    options, args = parser.parse_args()
    with open(options.config, 'r') as ymlfile:
        config = yaml.load(ymlfile)

    param = sanitise_parameter(config['parameter'])
    settings = sanitise_settings(config['settings'])

    K_tt, age_groups = load_age_mixing(config['data']['age_mixing_matrix_term'])
    K_hh, _ = load_age_mixing(config['data']['age_mixing_matrix_hol'])

    T, la_names = load_mobility_matrix(config['data']['mobility_matrix'])
    np.fill_diagonal(T, 0.)

    W = load_commute_volume(config['data']['commute_volume'], settings['inference_period'])['percent']

    N, n_names = load_population(config['data']['population_size'])

    K_tt = K_tt.astype(DTYPE)
    K_hh = K_hh.astype(DTYPE)
    W = W.to_numpy().astype(DTYPE)
    T = T.astype(DTYPE)
    N = N.astype(DTYPE)



    case_reports = pd.read_csv(config['data']['reported_cases'])
    case_reports['DateVal'] = pd.to_datetime(case_reports['DateVal'])
    case_reports = case_reports[case_reports['DateVal'] >= '2020-02-19']
    date_range = [case_reports['DateVal'].min(), case_reports['DateVal'].max()]
    y = case_reports['CumCases'].to_numpy()
    y_incr = np.round((y[1:] - y[:-1]) * 0.8)

    simulator = CovidUKODE(
        M_tt=K_tt,
        M_hh=K_hh,
        C=T,
        W=W,
        N=N,
        date_range=[date_range[0]-np.timedelta64(1,'D'), date_range[1]],
        holidays=settings['holiday'],
        lockdown=settings['lockdown'],
        time_step=int(settings['time_step']))

    seeding = seed_areas(N, n_names)  # Seed 40-44 age group, 30 seeds by popn size
    state_init = simulator.create_initial_state(init_matrix=seeding)

    def logp(par):
        p = param
        p['beta1'] = par[0]
        p['beta3'] = par[1]
        p['gamma'] = par[2]
        p['I0'] = par[3]
        p['r'] = par[4]
        beta_logp = tfd.Gamma(concentration=tf.constant(1., dtype=DTYPE), rate=tf.constant(1., dtype=DTYPE)).log_prob(p['beta1'])
        beta3_logp = tfd.Gamma(concentration=tf.constant(20., dtype=DTYPE),
                               rate=tf.constant(20., dtype=DTYPE)).log_prob(p['beta3'])
        gamma_logp = tfd.Gamma(concentration=tf.constant(100., dtype=DTYPE), rate=tf.constant(400., dtype=DTYPE)).log_prob(p['gamma'])
        I0_logp = tfd.Gamma(concentration=tf.constant(1.5, dtype=DTYPE), rate=tf.constant(0.05, dtype=DTYPE)).log_prob(p['I0'])
        r_logp = tfd.Gamma(concentration=tf.constant(0.1, dtype=DTYPE), rate=tf.constant(0.1, dtype=DTYPE)).log_prob(p['gamma'])
        t, sim, solve = simulator.simulate(p, state_init)
        y_logp = covid19uk_logp(y_incr, sim, 0.1, p['r'])
        logp = beta_logp + beta3_logp + gamma_logp + I0_logp + r_logp + tf.reduce_sum(y_logp)
        return logp

    def trace_fn(_, pkr):
      return (
          pkr.inner_results.log_accept_ratio,
          pkr.inner_results.accepted_results.target_log_prob,
          pkr.inner_results.accepted_results.step_size)


    unconstraining_bijector = [tfb.Exp()]
    initial_mcmc_state = np.array([0.05, 1.0, 0.25, 1.0, 50.], dtype=np.float64)  # beta1, gamma, I0
    print("Initial log likelihood:", logp(initial_mcmc_state))

    @tf.function(autograph=False, experimental_compile=True)
    def sample(n_samples, init_state, scale, num_burnin_steps=0):
        return tfp.mcmc.sample_chain(
            num_results=n_samples,
            num_burnin_steps=num_burnin_steps,
            current_state=init_state,
            kernel=tfp.mcmc.TransformedTransitionKernel(
                    inner_kernel=tfp.mcmc.RandomWalkMetropolis(
                        target_log_prob_fn=logp,
                        new_state_fn=random_walk_mvnorm_fn(scale)
                    ),
                    bijector=unconstraining_bijector),
            trace_fn=lambda _, pkr: pkr.inner_results.is_accepted)

    joint_posterior = tf.zeros([0] + list(initial_mcmc_state.shape), dtype=DTYPE)

    scale = np.diag([0.1, 0.1, 0.1, 0.1, 1.])
    overall_start = time.perf_counter()

    num_covariance_estimation_iterations = 50
    num_covariance_estimation_samples = 50
    num_final_samples = 10000
    start = time.perf_counter()
    for i in range(num_covariance_estimation_iterations):
        step_start = time.perf_counter()
        samples, results = sample(num_covariance_estimation_samples,
                                  initial_mcmc_state,
                                  scale)
        step_end = time.perf_counter()
        print(f'{i} time {step_end - step_start}')
        print("Acceptance: ", results.numpy().mean())
        joint_posterior = tf.concat([joint_posterior, samples], axis=0)
        cov = tfp.stats.covariance(tf.math.log(joint_posterior))
        print(cov.numpy())
        scale = cov * 2.38**2 / joint_posterior.shape[1]
        initial_mcmc_state = joint_posterior[-1, :]

    step_start = time.perf_counter()
    samples, results = sample(num_final_samples,
                              init_state=joint_posterior[-1, :], scale=scale,)
    joint_posterior = tf.concat([joint_posterior, samples], axis=0)
    step_end = time.perf_counter()
    print(f'Sampling step time {step_end - step_start}')
    end = time.perf_counter()
    print(f"Simulation complete in {end-start} seconds")
    print("Acceptance: ", np.mean(results.numpy()))
    print(tfp.stats.covariance(tf.math.log(joint_posterior)))

    fig, ax = plt.subplots(1, joint_posterior.shape[1])
    for i in range(joint_posterior.shape[1]):
        ax[i].plot(joint_posterior[:, i])

    plt.show()
    print(f"Posterior mean: {np.mean(joint_posterior, axis=0)}")

    with open('pi_beta_2020-03-29.pkl', 'wb') as f:
        pkl.dump(joint_posterior, f)
