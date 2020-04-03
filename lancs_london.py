"""Compares incidence in London to Lancashire"""

import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd

from covid.pydata import collapse_pop

def plot_lancs_v_london(pred_filename):

    file = h5py.File(pred_filename, 'r')
    pop = collapse_pop('data/c2019modagepop.csv')

    utlas = {'Lancashire': 'E10000017',
             'Cumbria': 'E10000006',
             'London': 'E09000001,E09000033'}

    pred = file['prediction']

    def f(utla_code):
        idx = pop.index.get_level_values(0) == utla_code
        pop_ = pop.loc[idx].sum().to_numpy()
        pred_ = pred[..., idx, 1:3].sum(axis=-1).sum(axis=-1)

        q = lambda x: np.percentile(x, q=[2.5, 97.5], axis=0)
        limits = np.percentile(pred_, q=[2.5, 97.5], axis=0) / pop_ * 10000
        mean = pred_.mean(axis=0) / pop_ * 10000
        return np.array([limits[0], mean, limits[1]])

    dates = pd.to_datetime(file['date'][:].astype(str))

    fig = plt.figure()
    ax = fig.gca()

    for k, v in utlas.items():
        x = f(v)
        ax.fill_between(dates, x[0], x[2], color='lightgray', alpha=0.7)
        ax.plot(dates, x[1], label=k)
    plt.legend()
    plt.ylabel('Prevalence per 10000 people')
    fig.autofmt_xdate()

    file.close()

