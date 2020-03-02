import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tode = tfp.math.ode


popsize = 2500
state = np.array([np.full([popsize], 999.),
                  np.full([popsize], 0.),
                  np.full([popsize], 1.),
                  np.full([popsize], 0.)], dtype=np.float64)

K = np.random.uniform(size=[popsize, popsize])

param = {'beta': 0.0002, 'nu': 0.14, 'gamma': 0.14}



def h(t, state):
    print(state)
    state = tf.unstack(state, axis=0)
    S, E, I, R = state

    infec_rate = param['beta'] * S * tf.linalg.matvec(K, I)
    dS = -infec_rate
    dE = infec_rate - param['nu'] * E
    dI = param['nu'] * E - param['gamma'] * I
    dR = param['gamma'] * I

    df = tf.stack([dS, dE, dI, dR])
    return df

@tf.function
def solve_ode(rates, t_init, state_init, t):
    return tode.DormandPrince(first_step_size=1., max_num_steps=5000).solve(rates, t_init, state_init, solution_times=t)

solution_times = np.arange(0., 365., 1.)

print("Running...", flush=True)
result = solve_ode(rates=h, t_init=0., state_init=state, t=solution_times)
print("Done")
print(result)