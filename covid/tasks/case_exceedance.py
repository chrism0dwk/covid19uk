"""Calculates case exceedance probabilities"""

import numpy as np
import pickle as pkl
import pandas as pd


def case_exceedance(input_files, lag):
    """Calculates case exceedance probabilities,
       i.e. Pr(pred[lag:] < observed[lag:])

    :param input_files: [data pickle, prediction pickle]
    :param lag: the lag for which to calculate the exceedance
    """

    with open(input_files[0], "rb") as f:
        data = pkl.load(f)

    with open(input_files[1], "rb") as f:
        prediction = pkl.load(f)

    modelled_cases = np.sum(prediction[..., :lag, -1], axis=-1)
    observed_cases = np.sum(data["cases"].to_numpy()[:, -lag:], axis=-1)
    exceedance = np.mean(modelled_cases < observed_cases, axis=0)

    df = pd.Series(
        exceedance,
        index=pd.Index(data["locations"]["lad19cd"], name="location"),
    )

    return df
