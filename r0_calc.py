import numpy as np
from scipy import optimize as opt

from covid.rdata import *
from covid.model import CovidUKODE

if __name__=='__main__':

    K, _ = load_age_mixing('data/polymod_normal_df.rds')
    T, _ = load_mobility_matrix('data/movement.rds')
    np.fill_diagonal(T, 0.)
    N, _ = load_population('data/pop.rds')

    model = CovidUKODE(K, T, N)

    print("R0 (beta=0.036) = ",  model.eval_R0({'beta1': 0.036, 'beta2': 0.33, 'gamma': 0.25}))
    R0 = 2.4

    def opt_fn(beta, R0):
        r0, i = model.eval_R0({'beta1': beta, 'beta2': 0.33, 'gamma': 0.25}, 1e-9)
        return (r0 - R0)**2

    res = opt.minimize_scalar(opt_fn, args=R0)
    print(res['x'])
