"""Function responsible for assembling all data required
   to instantiate the COVID19 model"""

import os
from covid.model_spec import gather_data


def assemble_data(filename, config):

    constant_data, observations = gather_data(config)
    if os.path.exists(filename):
        mode = "a"
    else:
        mode = "w"
    constant_data.to_netcdf(filename, group="constant_data", mode=mode)
    observations.to_netcdf(filename, group="observations", mode="a")


if __name__ == "__main__":

    from argparse import ArgumentParser
    import yaml

    parser = ArgumentParser(description="Bundle data into a pickled dictionary")
    parser.add_argument("config_file", help="Global config file")
    parser.add_argument("output_file", help="InferenceData file")
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        global_config = yaml.load(f, Loader=yaml.FullLoader)

    assemble_data(args.output_file, global_config["ProcessData"])
