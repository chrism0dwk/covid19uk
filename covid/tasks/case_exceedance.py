"""Calculates case exceedance probabilities"""

import numpy as np
import pickle as pkl
import xarray


def case_exceedance(input_files, lag):
    """Calculates case exceedance probabilities,
       i.e. Pr(pred[lag:] < observed[lag:])

    :param input_files: [data pickle, prediction pickle]
    :param lag: the lag for which to calculate the exceedance
    """
    data_file, prediction_file = input_files

    with open(data_file, "rb") as f:
        data = pkl.load(f)

    prediction = xarray.open_dataset(prediction_file)["events"]

    modelled_cases = np.sum(prediction[..., :lag, -1], axis=-1)
    observed_cases = np.sum(data["cases"][:, -lag:], axis=-1)
    if observed_cases.dims[0] == "lad19cd":
        observed_cases = observed_cases.rename({"lad19cd": "location"})
    exceedance = np.mean(modelled_cases < observed_cases, axis=0)

    return exceedance


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Calculates case exceedance probabilities"
    )
    parser.add_argument("data_file", type=str)
    parser.add_argument("prediction_file", type=str)
    parser.add_argument(
        "-l",
        "--lag",
        type=int,
        help="The lag for which to calculate exceedance",
        default=7,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="The output csv",
        default="exceedance.csv",
    )
    args = parser.parse_args()

    df = case_exceedance([args.data_file, args.prediction_file], args.lag)
    df.to_csv(args.output)
