"""A Ruffus-ised pipeline for COVID-19 analysis"""

import os
from os.path import expandvars
import warnings
import yaml
import datetime
import s3fs
import ruffus as rf

from covid.ruffus_pipeline import run_pipeline


def _import_global_config(config_file):

    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config


if __name__ == "__main__":

    # Ruffus wrapper around argparse used to give us ruffus
    # cmd line switches as well as our own config
    argparser = rf.cmdline.get_argparse(description="COVID-19 pipeline")
    data_args = argparser.add_argument_group(
        "Data options", "Options controlling input data"
    )

    data_args.add_argument(
        "-c",
        "--config",
        type=str,
        help="global configuration file",
        required=True,
    )
    data_args.add_argument(
        "-r",
        "--results-directory",
        type=str,
        help="pipeline results directory",
        required=True,
    )
    data_args.add_argument(
        "--date-range",
        type=lambda s: datetime.datetime.strptime(s, "%Y-%m-%d"),
        nargs=2,
        help="Date range [low high)",
        metavar="ISO6801",
    )
    data_args.add_argument(
        "--reported-cases", type=str, help="Path to case file"
    )
    data_args.add_argument(
        "--commute-volume", type=str, help="Path to commute volume file"
    )
    data_args.add_argument(
        "--case-date-type",
        type=str,
        help="Case date type (specimen | report)",
        choices=["specimen", "report"],
    )
    data_args.add_argument(
        "--pillar", type=str, help="Pillar", choices=["both", "1", "2"]
    )
    data_args.add_argument(
        "--aws", action='store_true', help="Push to AWS"
        )

    cli_options = argparser.parse_args()
    global_config = _import_global_config(cli_options.config)

    if cli_options.date_range is not None:
        global_config["ProcessData"]["date_range"][0] = cli_options.date_range[
            0
        ]
        global_config["ProcessData"]["date_range"][1] = cli_options.date_range[
            1
        ]

    if cli_options.reported_cases is not None:
        global_config["ProcessData"]["CasesData"]["address"] = expandvars(
            cli_options.reported_cases
        )

    if cli_options.commute_volume is not None:
        global_config["ProcessData"]["commute_volume"] = expandvars(
            cli_options.commute_volume
        )

    if cli_options.case_date_type is not None:
        global_config["ProcessData"][
            "case_date_type"
        ] = cli_options.case_date_type

    if cli_options.pillar is not None:
        opts = {
            "both": ["Pillar 1", "Pillar 2"],
            "1": ["Pillar 1"],
            "2": ["Pillar 2"],
        }
        global_config["ProcessData"]["CasesData"]["pillars"] = opts[
            cli_options.pillar
        ]

    run_pipeline(global_config, cli_options.results_directory, cli_options)

    if cli_options.aws is True:
        bucket_name = global_config['AWSS3']['bucket']
        obj_name = os.path.split(cli_options.results_directory)[1]
        obj_path = f"{bucket_name}/{obj_name}"
        s3 = s3fs.S3FileSystem(profile=global_config["AWSS3"]["profile"])
        if not s3.exists(obj_path):
            s3.put(cli_options.results_directory, obj_path, recursive=True)
        else:
            warnings.warn(f"Path '{obj_path}' already exists, not uploading.")
