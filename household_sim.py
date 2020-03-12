"""Household simulation for SPI-M"""
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from covid.model import CovidUK

# Household size distribution in the UK, taken from ONS, 2019
hh_size_distr = pd.DataFrame({'size': [1, 2, 3, 4, 5, 6], 'freq': [6608, 8094, 3918, 3469, 1186, 463]})


# Draw households
def draw_households(n: int, dist: pd.DataFrame) -> np.array:
    probs = dist['freq'] / np.sum(dist['freq'])
    hh_sizes = np.random.choice(dist['size'], n, p=probs)
    #hh_vec = np.repeat(np.arange(hh_sizes.shape[0]), hh_sizes)
    return hh_sizes


# Simulator
class HHstochastic(CovidUK):

    def __init__(self, N):
        self.stoichiometry = tf.constant([[-1, 1,  0,  0, 0],  # S->E
                                          [0, -1,  1,  0, 0],  # E->I
                                          [0,  0, -1,  1, 0],  # I->D
                                          [0,  0, -1,  0, 1],  # I->R
                                          [0,  0,  0, -1, 1]], dtype=tf.float64)  # D->R
        self.N = N
        self.popsize = tf.reduce_sum(N)

    def rules(self, state):
        tf.

    def h(self, state):
        state = tf.unstack(state, axis=0)
        S, E, I, D, R = state

        eye = tf.eye(I.shape[1], dtype=tf.float64)

        infec_rate = tf.linalg.tensordot(I, eye, axes=[[1], [1]])
        infec_rate /= self.N
        infec_rate *= self.param['omega']
        infec_rate += tf.reduce_sum(I, axis=1, keepdims=True) / self.popsize
        infec_rate *= self.param['beta']

        hazard_rates = tf.stack([
            infec_rate,  # S->E
            tf.constant(np.full(S.shape, self.param['nu']), dtype=tf.float64),  # E->I
            tf.constant(np.full(S.shape, self.param['alpha']), dtype=tf.float64),  # I->D
            self.param['rho'] * D,  # S->R
            tf.constant(np.full(S.shape, self.param['gamma']), dtype=tf.float64)  # I->R
        ])
        return tf.constant([0, 1, 2, 2, 3]), hazard_rates


if __name__=='__main__':
    hh = draw_households(20, hh_size_distr)
    nsim = 2

    param = {'beta': 0.3,     # Baseline infection
             'omega': 1.5,    # Additional transmission in house
             'nu': 0.25,      # Incubation rate
             'alpha': 0.5,    # Prodromal
             'gamma': 0.5,    # Symptomatic
             'rho': 10e6}   # Isolation for whole house

    model = HHstochastic(hh.astype(np.float64))
    init_state = np.stack([np.broadcast_to(hh, [nsim, hh.shape[0]]),  # S
                           np.zeros([nsim, hh.shape[0]]),  # E
                           np.zeros([nsim, hh.shape[0]]),  # I
                           np.zeros([nsim, hh.shape[0]]),  # D
                           np.zeros([nsim, hh.shape[0]])]).astype(np.float64)  # R
    init_state[2, :, 0] = 1.
    init_state[0, :, 0] = init_state[0, :, 0] - 1.

    start = time.perf_counter()
    t, sim = model.sample(init_state, [0., 5.], param)
    end = time.perf_counter()
    print(f"Completed in {end-start} seconds")

    r = tf.reduce_sum(sim[:, 2, :, :], axis=2)

    plt.plot(t, r, 'r-', alpha=0.2)
    plt.show()
