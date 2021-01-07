"""A Ruffus-ised pipeline for COVID-19 analysis"""

import os
import yaml
import ruffus as rf

from covid.model_spec import gather_data
from covid.tasks import (
    inference,
    thin_posterior,
    next_generation_matrix,
    overall_rt,
    predict,
    geopackage,
    #    lancs_spreadsheet,
)


def import_global_config(args):

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config


def join_and_expand(path1, path2):
    return os.path.expand(os.path.join(path1, path2))


if __name__ == "__main__":

    argparser = rf.cmdline.get_argparse(description="COVID-19 pipeline")
    argparser.add_argument(
        "-c", "--config", type=str, help="global configuration file"
    )
    argparser.add_argument(
        "-r", "--results-directory", type=str, help="pipeline results directory"
    )
    cli_options = argparser.parse_args()
    global_config = import_global_config(cli_options.config)

    # Output paths
    BASEDIR = os.path.expandvars(cli_options.results_directory)
    p = lambda fn: os.path.join(BASEDIR, fn)

    # Pipeline starts here
    @rf.mkdir(BASEDIR)
    @rf.originate(p("config.yaml"), global_config)
    def save_config(output_file, global_config):
        with open(output_file, "w"):
            yaml.dump(global_config)

    @rf.transform(
        save_config,
        rf.formatter(),
        p("pipeline_data.pkl"),
        global_config,
    )
    def process_data(input_file, output_file, global_config):
        data = gather_data(global_config)
        with open(output_file, "wb") as f:
            pkl.dump(data, f)

    rf.transform(
        inference,
        process_data,
        rf.formatter(),
        p("posterior.hd5"),
        global_config["Inference"],
    )

    rf.transform(
        thin_posterior,
        input=inference,
        filter=rf.formatter(),
        output=p("thin_samples.pkl"),
        indices=range(6000, 10000, 10),
    )

    # Rt related steps
    rf.transform(
        next_generation_matrix,
        input=[process_data, thin_posterior],
        filter=rf.formatter(),
        output=p("ngm.pkl"),
    )

    rf.transform(
        overall_rt,
        input=next_generation_matrix,
        filter=rf.formatter(),
        output=p("national_rt.xlsx"),
    )

    # In-sample prediction
    @rf.transform(
        input=[process_data, thin_posterior],
        filter=rf.formatter(),
        output=p("insample7.pkl"),
    )
    def insample7(input_files, output_file):
        return predict(
            data=input_files[0],
            posterior_samples=input_files[1],
            output_file=output_file,
            timespan=[-7, -1],
        )

    @rf.transform(
        input=[process_data, thin_posterior],
        filter=rf.formatter(),
        output=p("insample14.pkl"),
    )
    def insample14(input_files, output_file):
        return predict(
            data=input_files[0],
            posterior_samples=input_files[1],
            output_file=output_file,
            initial_step=-14,
            num_steps=14,
        )

    # Medium-term prediction
    @rf.transform(
        input=[process_data, thin_posterior],
        filter=rf.formatter(),
        output=p("medium_term.pkl"),
    )
    def medium_term(input_files, output_file):
        return predict(
            data=input_files[0],
            posterior_samples=input_files[1],
            output_file=output_file,
            initial_step=-1,
            num_steps=56,
        )

    # Assemble
    rf.transform(
        geopackage,
        input=[next_generation_matrix, insample7, medium_term],
        filter=rf.formatter(),
        output=p("prediction.gpkg"),
        config=global_config["Geopackage"],
    )

    rf.cmdline(cli_options)
