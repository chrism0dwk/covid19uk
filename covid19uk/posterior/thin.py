"""Thin posterior"""

import h5py
import pickle as pkl


def thin_posterior(input_file, output_file, config):

    idx = slice(config["start"], config["end"], config["by"])

    f = h5py.File(input_file, "r", rdcc_nbytes=1024 ** 3, rdcc_nslots=1e6)

    output_dict = {k: v[idx] for k, v in f["samples"].items()}
    output_dict["initial_state"] = f["initial_state"][:]
    f.close()

    with open(output_file, "wb") as f:
        pkl.dump(
            output_dict,
            f,
        )


if __name__ == "__main__":

    import yaml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, help="Configuration file", required=True
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Output pkl file", required=True
    )
    parser.add_argument("samples", type=str, help="Posterior HDF5 file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print("Config: ", config["ThinPosterior"])
    thin_posterior(args.samples, args.output, config["ThinPosterior"])
