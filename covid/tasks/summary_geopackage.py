"""Summarises posterior distribution into a geopackage"""

import pickle as pkl
import xarray
import pandas as pd
import geopandas as gp


def _tier_enum(design_matrix):
    """Turns a factor variable design matrix into
    an enumerated column"""
    df = design_matrix[-1].to_dataframe()[["value"]]
    df = df[df["value"] == 1.0].reset_index()
    return df["alert_level"]


def summary_geopackage(input_files, output_file, config):
    """Creates a summary geopackage file

    :param input_files: a list of data file names [data pkl,
                                                   next_generation_matrix,
                                                   insample7,
                                                   insample14,
                                                   medium_term]
    :param output_file: the output geopackage file
    :param config: SummaryGeopackage configuration information
    """

    # Read in the first input file
    data = xarray.open_dataset(input_files.pop(0), group="constant_data")

    # Load and filter geopackage
    geo = gp.read_file(config["base_geopackage"], layer=config["base_layer"])
    geo = geo[geo["lad19cd"].isin(data.coords["location"])]
    geo = geo.sort_values(by="lad19cd")

    # Dump data into the geopackage
    while len(input_files) > 0:
        fn = input_files.pop()
        print(f"Collating {fn}")
        try:
            columns = pd.read_csv(fn, index_col="location")
        except ValueError as e:
            raise ValueError(f"Error reading file '{fn}': {e}")

        geo = geo.merge(
            columns, how="left", left_on="lad19cd", right_index=True
        )

    geo.to_file(output_file, driver="GPKG")
