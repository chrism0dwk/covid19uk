import optparse
import numpy as np
import yaml
import time
import matplotlib.pyplot as plt

from covid.model import CovidUKODE
from covid.rdata import *


def sanitise_parameter(par_dict):
    """Sanitises a dictionary of parameters"""
    par = ['beta','nu','gamma']
    d = {key: np.float64(par_dict[key]) for key in par}
    return d


def sanitise_settings(par_dict):
    d = {'start': np.float64(par_dict['start']),
         'end': np.float64(par_dict['end']),
         'time_step': np.float64(par_dict['time_step'])}
    return d


if __name__ == '__main__':

    parser = optparse.OptionParser()
    parser.add_option("--config", "-c", dest="config",
                      help="configuration file")
    options, args = parser.parse_args()
    with open(options.config, 'r') as ymlfile:
        config = yaml.load(ymlfile)

    K = load_age_mixing(config['data']['age_mixing_matrix']).to_numpy(dtype=np.float32)
    T = load_mobility_matrix(config['data']['mobility_matrix']).to_numpy(dtype=np.float32)
    np.fill_diagonal(T, 0.)
    N = load_population(config['data']['population_size'])['n'].to_numpy(dtype=np.float32)

    param = sanitise_parameter(config['parameter'])
    settings = sanitise_settings(config['settings'])

    model = CovidUKODE(K, T, N, T.shape[0])

    state_init = model.create_initial_state()
    start = time.perf_counter()
    t, sim = model.simulate(param, state_init,
                            settings['start'], settings['end'],
                            settings['time_step'])
    end = time.perf_counter()
    print(f"Complete in {end-start} seconds")
    plt.plot(t, sim[:, 2, 0], 'r-')
    plt.show()