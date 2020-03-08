import numpy as np

from covid.rdata import *
from covid.model import CovidUKODE

K = load_age_mixing('data/polymod_normal_df.rds').to_numpy(dtype=np.float32)
T = load_mobility_matrix('data/movement.rds').to_numpy(dtype=np.float32)
np.fill_diagonal(T, 0.)
N = load_population('data/pop.rds')['n'].to_numpy(dtype=np.float32)

model = CovidUKODE(K, T, N, T.shape[0])

