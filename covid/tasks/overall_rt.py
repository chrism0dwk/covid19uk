"""Calculates overall Rt given a posterior next generation matix"""

import numpy as np
import xarray
import pandas as pd

from covid.summary import (
    rayleigh_quotient,
    power_iteration,
)


def overall_rt(next_generation_matrix, output_file):

    ngms = xarray.open_dataset(next_generation_matrix)["ngm"]
    b, _ = power_iteration(ngms)
    rt = rayleigh_quotient(ngms, b)
    q = np.arange(0.05, 1.0, 0.05)
    rt_quantiles = pd.DataFrame(
        {"Rt": np.quantile(rt, q, axis=-1)}, index=q
    ).T.to_excel(output_file)


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "input_file",
        description="The input .pkl file containing the next generation matrix",
    )
    parser.add_argument(
        "output_file", description="The name of the output .xlsx file"
    )

    args = parser.parse_args()
    overall_rt(args.input_file, args.output_file)
