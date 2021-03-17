"""Represents the analytic pipeline as a ruffus chain"""

import os
import yaml
from datetime import datetime
from uuid import uuid1
import json
import netCDF4 as nc
import pandas as pd
import ruffus as rf


from covid.tasks import (
    assemble_data,
    mcmc,
    thin_posterior,
    reproduction_number,
    overall_rt,
    predict,
    summarize,
    within_between,
    case_exceedance,
    summary_geopackage,
    insample_predictive_timeseries,
    summary_longformat,
)

__all__ = ["run_pipeline"]


def _make_append_work_dir(work_dir):
    return lambda filename: os.path.expandvars(os.path.join(work_dir, filename))


def _create_metadata(config):
    return dict(
        pipeline_id=uuid1().hex,
        created_at=str(datetime.now()),
        inference_library="GEM",
        inference_library_version="0.1.alpha0",
        pipeline_config=json.dumps(config, default=str),
    )


def _create_nc_file(output_file, meta_data_dict):
    nc_file = nc.Dataset(output_file, "w", format="NETCDF4")
    for k, v in meta_data_dict.items():
        setattr(nc_file, k, v)
    nc_file.close()


def run_pipeline(global_config, results_directory, cli_options):

    wd = _make_append_work_dir(results_directory)

    pipeline_meta = _create_metadata(global_config)

    # Pipeline starts here
    @rf.mkdir(results_directory)
    @rf.originate(wd("config.yaml"), global_config)
    def save_config(output_file, config):
        with open(output_file, "w") as f:
            yaml.dump(config, f)

    @rf.follows(save_config)
    @rf.originate(wd("inferencedata.nc"), global_config)
    def process_data(output_file, config):

        _create_nc_file(output_file, pipeline_meta)
        assemble_data(output_file, config["ProcessData"])

    @rf.transform(
        process_data,
        rf.formatter(),
        wd("posterior.hd5"),
        global_config,
    )
    def run_mcmc(input_file, output_file, config):
        mcmc(input_file, output_file, config["Mcmc"])

    @rf.transform(
        input=run_mcmc,
        filter=rf.formatter(),
        output=wd("thin_samples.pkl"),
        extras=[global_config],
    )
    def thin_samples(input_file, output_file, config):
        thin_posterior(input_file, output_file, config["ThinPosterior"])

    # Rt related steps
    rf.transform(
        input=[[process_data, thin_samples]],
        filter=rf.formatter(),
        output=wd("reproduction_number.nc"),
    )(reproduction_number)

    rf.transform(
        input=reproduction_number,
        filter=rf.formatter(),
        output=wd("national_rt.xlsx"),
    )(overall_rt)

    # In-sample prediction
    @rf.transform(
        input=[[process_data, thin_samples]],
        filter=rf.formatter(),
        output=wd("insample7.nc"),
    )
    def insample7(input_files, output_file):
        predict(
            data=input_files[0],
            posterior_samples=input_files[1],
            output_file=output_file,
            initial_step=-7,
            num_steps=28,
        )

    @rf.transform(
        input=[[process_data, thin_samples]],
        filter=rf.formatter(),
        output=wd("insample14.nc"),
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
        output=wd("medium_term.nc"),
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
        input=reproduction_number,
        filter=rf.formatter(),
        output=wd("rt_summary.csv"),
    )(summarize.rt)

    rf.transform(
        input=medium_term,
        filter=rf.formatter(),
        output=wd("infec_incidence_summary.csv"),
    )(summarize.infec_incidence)

    rf.transform(
        input=[[process_data, medium_term]],
        filter=rf.formatter(),
        output=wd("prevalence_summary.csv"),
    )(summarize.prevalence)

    rf.transform(
        input=[[process_data, thin_samples]],
        filter=rf.formatter(),
        output=wd("within_between_summary.csv"),
    )(within_between)

    @rf.transform(
        input=[[process_data, insample7, insample14]],
        filter=rf.formatter(),
        output=wd("exceedance_summary.csv"),
    )
    def exceedance(input_files, output_file):
        exceed7 = case_exceedance((input_files[0], input_files[1]), 7)
        exceed14 = case_exceedance((input_files[0], input_files[2]), 14)
        df = pd.DataFrame(
            {"Pr(pred<obs)_7": exceed7, "Pr(pred<obs)_14": exceed14},
            index=exceed7.coords["location"],
        )
        df.to_csv(output_file)

    # Plot in-sample
    @rf.transform(
        input=[insample7, insample14],
        filter=rf.formatter(".+/insample(?P<LAG>\d+).nc"),
        add_inputs=rf.add_inputs(process_data),
        output="{path[0]}/insample_plots{LAG[0]}",
        extras=["{LAG[0]}"],
    )
    def plot_insample_predictive_timeseries(input_files, output_dir, lag):
        insample_predictive_timeseries(input_files, output_dir, lag)

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
        wd("prediction.gpkg"),
        global_config["Geopackage"],
    )(summary_geopackage)

    rf.cmdline.run(cli_options)

    # DSTL Summary
    rf.transform(
        [
            [
                process_data,
                insample7,
                insample14,
                medium_term,
                reproduction_number,
            ]
        ],
        rf.formatter(),
        wd("summary_longformat.xlsx"),
    )(summary_longformat)

    rf.cmdline.run(cli_options)
