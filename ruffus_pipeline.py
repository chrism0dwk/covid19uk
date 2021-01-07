"""A Ruffus-ised pipeline for COVID-19 analysis"""

import os
import yaml
import pandas as pd
import ruffus as rf

from covid.tasks import (
    assemble_data,
    mcmc,
    thin_posterior,
    next_generation_matrix,
    overall_rt,
    predict,
    summarize,
    within_between,
    case_exceedance,
    summary_geopackage,
    # lancs_spreadsheet,
)


def import_global_config(config_file):

    with open(config_file, "r") as f:
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

    def work_dir(fn):
        return os.path.join(BASEDIR, fn)

    # Pipeline starts here
    @rf.mkdir(BASEDIR)
    @rf.originate(work_dir("config.yaml"), global_config)
    def save_config(output_file, config):
        with open(output_file, "w") as f:
            yaml.dump(config, f)

    @rf.transform(
        save_config,
        rf.formatter(),
        work_dir("pipeline_data.pkl"),
        global_config,
    )
    def process_data(input_file, output_file, config):
        assemble_data(output_file, config["ProcessData"])

    @rf.transform(
        process_data,
        rf.formatter(),
        work_dir("posterior.hd5"),
        global_config,
    )
    def run_mcmc(input_file, output_file, config):
        mcmc(input_file, output_file, config["Mcmc"])

    @rf.transform(
        input=run_mcmc,
        filter=rf.formatter(),
        output=work_dir("thin_samples.pkl"),
    )
    def thin_samples(input_file, output_file):
        thin_posterior(input_file, output_file, range(100))

    # Rt related steps
    rf.transform(
        input=[[process_data, thin_samples]],
        filter=rf.formatter(),
        output=work_dir("ngm.pkl"),
    )(next_generation_matrix)

    rf.transform(
        input=next_generation_matrix,
        filter=rf.formatter(),
        output=work_dir("national_rt.xlsx"),
    )(overall_rt)

    # In-sample prediction
    @rf.transform(
        input=[[process_data, thin_samples]],
        filter=rf.formatter(),
        output=work_dir("insample7.pkl"),
    )
    def insample7(input_files, output_file):
        predict(
            data=input_files[0],
            posterior_samples=input_files[1],
            output_file=output_file,
            initial_step=-8,
            num_steps=28,
        )

    @rf.transform(
        input=[[process_data, thin_samples]],
        filter=rf.formatter(),
        output=work_dir("insample14.pkl"),
    )
    def insample14(input_files, output_file):
        return predict(
            data=input_files[0],
            posterior_samples=input_files[1],
            output_file=output_file,
            initial_step=-14,
            num_steps=28,
        )

    # Medium-term prediction
    @rf.transform(
        input=[[process_data, thin_samples]],
        filter=rf.formatter(),
        output=work_dir("medium_term.pkl"),
    )
    def medium_term(input_files, output_file):
        return predict(
            data=input_files[0],
            posterior_samples=input_files[1],
            output_file=output_file,
            initial_step=-1,
            num_steps=61,
        )

    # Summarisation
    rf.transform(
        input=next_generation_matrix,
        filter=rf.formatter(),
        output=work_dir("rt_summary.csv"),
    )(summarize.rt)

    rf.transform(
        input=medium_term,
        filter=rf.formatter(),
        output=work_dir("infec_incidence_summary.csv"),
    )(summarize.infec_incidence)

    rf.transform(
        input=[[process_data, thin_samples, medium_term]],
        filter=rf.formatter(),
        output=work_dir("prevalence_summary.csv"),
    )(summarize.prevalence)

    rf.transform(
        input=[[process_data, thin_samples]],
        filter=rf.formatter(),
        output=work_dir("within_between_summary.csv"),
    )(within_between)

    @rf.transform(
        input=[[process_data, insample7, insample14]],
        filter=rf.formatter(),
        output=work_dir("exceedance_summary.csv"),
    )
    def exceedance(input_files, output_file):
        exceed7 = case_exceedance((input_files[0], input_files[1]), 7)
        exceed14 = case_exceedance((input_files[0], input_files[2]), 14)
        df = pd.DataFrame(
            {"Pr(pred<obs)_7": exceed7, "Pr(pred<obs)_14": exceed14}
        )
        df.to_csv(output_file)

    # @rf.transform(
    #     input=[[process_data, insample7, insample14, medium_term]],
    #     filter=rf.formatter(),
    #     output=work_dir("total_predictive_timeseries.pdf")
    # )(total_predictive_timeseries)

    # Geopackage
    rf.transform(
        [
            [
                process_data,
                summarize.rt,
                summarize.infec_incidence,
                summarize.prevalence,
                within_between,
                exceedance,
            ]
        ],
        rf.formatter(),
        work_dir("prediction.gpkg"),
        global_config["Geopackage"],
    )(summary_geopackage)

    rf.cmdline.run(cli_options)
