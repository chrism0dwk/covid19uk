"""Function responsible for assembling all data required
   to instantiate the COVID19 model"""


import pickle as pkl

from covid.model_spec import gather_data


def assemble_data(output_file, config):

    all_data = gather_data(config)
    with open(output_file, "wb") as f:
        pkl.dump(all_data, f)


if __name__ == "__main__":

    from argparse import ArgumentParser
    import yaml

    parser = ArgumentParser(description="Bundle data into a pickled dictionary")
    parser.add_argument("config_file", help="Global config file")
    parser.add_argument("output_file", help="Data bundle pkl file")
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        global_config = yaml.load(f, Loader=yaml.FullLoader)

    assemble_data(args.output_file, global_config["ProcessData"])
