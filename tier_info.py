"""Adds Tier info to geopackage"""

import os
import yaml
import h5py
import numpy as np
import geopandas as gp
from covid.cli_arg_parse import cli_args
import model_spec

DTYPE = model_spec.DTYPE

GIS_TEMPLATE = "data/UK2019mod_pop.gpkg"

if __name__ == "__main__":

    args = cli_args()

    # Get general config
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Load covariate data
    covar_data = model_spec.read_covariates(config)

    # Load geopackage
    geo = gp.read_file(
        os.path.join(
            config["output"]["results_dir"], config["output"]["geopackage"]
        )
    )
    geo = geo[geo["lad19cd"].str.startswith("E")]  # England only, for now.
    geo = geo.sort_values("lad19cd")

    tiers = covar_data["L"][-1].to_dataframe()[["value"]]
    tiers = tiers[tiers["value"] == 1.0].reset_index()
    geo["current_alert_level"] = tiers["alert_level"]
    geo.to_file(
        os.path.join(
            config["output"]["results_dir"], config["output"]["geopackage"]
        ),
        driver="GPKG",
    )
