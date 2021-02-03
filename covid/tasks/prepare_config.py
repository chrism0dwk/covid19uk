"""Prepare a config file for the COVID-19 pipeline"""

import sys
from pathlib import Path
import shutil
import numpy as np
import argparse
import yaml


def maybe_delete_dir(path):
    dirpath = Path(path)
    if dirpath.exists():
        shutil.rmtree(dirpath)

def maybe_make_dir(path):
    dirpath = Path(path)
    dirpath.mkdir(exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-o",
    "--output",
    type=str,
    default="stdout",
    help="Output file. If unspecified or `stdout`, writes to stdout.",
)
parser.add_argument(
    "--inference-period",
    type=np.datetime64,
    nargs=2,
    help="Date range [low high)",
    metavar="ISO6801",
)
parser.add_argument("--reported-cases", type=str, help="Path to case file")
parser.add_argument("--commute-volume", type=str, help="Path to commute volume file")
parser.add_argument(
    "--case-date-type",
    type=str,
    help="Case date type (specimen | report)",
    choices=["specimen", "report"],
)
parser.add_argument("--pillar", type=str, help="Pillar", choices=["both", "1", "2"])
parser.add_argument("--results-dir", type=str, help="Results directory")
parser.add_argument("--delete", type=bool, default=False, help="Delete an existing results dir before writing")
parser.add_argument("config_template", type=str, help="Config file template")

args = parser.parse_args()
template_file = args.config_template
output_file = args.output

with open(template_file, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

for key in ["reported_cases", "commute_volume", "case_date_type", "pillar"]:
    val = getattr(args, key)
    if val is not None:
        config["data"][key] = val

if args.inference_period is not None:
    config["Global"]["inference_period"] = [str(x) for x in args.inference_period]

for key in ["results_dir"]:
    val = getattr(args, key)
    if val is not None:
        config["output"][key] = val

if args.delete is True:
    maybe_delete_dir(config["output"]["results_dir"])
maybe_make_dir(config["output"]["results_dir"])

if args.output == "stdout":
    print(yaml.dump(config))
else:
    with open(args.output, "w") as f:
        yaml.dump(config, f)
    print(args.output)

sys.exit(0)
