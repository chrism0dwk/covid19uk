import tensorboard as tb
import numpy as np

from covid.model import Homogeneous

model = Homogeneous()
popsize = 10
state = np.array([np.full([popsize], 999.),
                  np.full([popsize], 0.),
                  np.full([popsize], 1.),
                  np.full([popsize], 0.)], dtype=np.float32)
print(state)
print("Running...", flush=True, sep='')
t, sim_state = model.sample(state,
                            [0., 365.],
                            {'beta': 0.2, 'nu': 0.14, 'gamma':0.14})
print("Done", flush=True)
