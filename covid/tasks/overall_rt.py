"""Calculates overall Rt given a posterior next generation matix"""

import numpy as np
import xarray
import pandas as pd

from covid.summary import (
    rayleigh_quotient,
    power_iteration,
)


def overall_rt(inference_data, output_file):

    r_t = xarray.open_dataset(inference_data, group="posterior_predictive")[
        "R_t"
    ]

    q = np.arange(0.05, 1.0, 0.05)
    quantiles = r_t.isel(time=-1).quantile(q=q)
    quantiles.to_dataframe().T.to_excel(output_file)
    # pd.DataFrame({"Rt": np.quantile(r_t, q, axis=-1)}, index=q).T.to_excel(
    #     output_file
    # )


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "input_file",
        type=str,
        help="The input .pkl file containing the next generation matrix",
    )
    parser.add_argument(
        "output_file", type=str, help="The name of the output .xlsx file"
    )

    args = parser.parse_args()
    overall_rt(args.input_file, args.output_file)
