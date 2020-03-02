import tensorflow as tf
import numpy as np

from covid.model import Homogeneous

sw = tf.summary.create_file_writer('tf_log')

model = Homogeneous()
popsize = 2500
state = np.array([np.full([popsize], 999.),
                  np.full([popsize], 0.),
                  np.full([popsize], 1.),
                  np.full([popsize], 0.)], dtype=np.float32)

print("Running...", flush=True, sep='')
tf.summary.trace_on(graph=True, profiler=True)

t, sim_state = model.sample(state,
                            [0., 10.],
                            {'beta': 0.2, 'nu': 0.14, 'gamma':0.14})
print("Done", flush=True)
print(sim_state)

with sw.as_default():
    tf.summary.trace_export('profile', step=0, profiler_outdir='tf_log')

sw.flush()
sw.close()
