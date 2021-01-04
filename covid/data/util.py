"""Utility functions for COVID19 UK data"""

import os
import re
import datetime
import numpy as np
import pandas as pd


def prependDate(filename):
    now = datetime.now()  # current date and time
    date_time = now.strftime("%Y-%m-%d")
    return date_time + "_" + filename


def prependID(filename, config):
    return config["Global"]["prependID_Str"] + "_" + filename


def format_input_filename(filename, config):
    # prepend with a set string
    # to load a specific date, this should be in the string
    p, f = os.path.split(filename)
    if config["Global"]["prependID"]:
        f = prependID(f, config)
    filename = p + "/" + f
    return filename


def format_output_filename(filename, config):
    p, f = os.path.split(filename)
    if config["Global"]["prependID"]:
        f = prependID(f, config)
    if config["Global"]["prependDate"]:
        f = prependDate(f)
    filename = p + "/" + f
    return filename


def merge_lad_codes(lad19cd):
    merging = {
        "E06000052": "E06000052,E06000053",  # City of London & Westminster
        "E06000053": "E06000052,E06000053",  # City of London & Westminster
        "E09000001": "E09000001,E09000033",  # Cornwall & Isles of Scilly
        "E09000033": "E09000001,E09000033",  # Cornwall & Isles of Scilly
    }
    lad19cd = lad19cd.apply(lambda x: merging[x] if x in merging.keys() else x)

    return lad19cd


def merge_lad_values(df):
    df = df.groupby("lad19cd").sum().reset_index()
    return df


def get_date_low_high(config):
    if "dates" in config:
        low = config["dates"]["low"]
        high = config["dates"]["high"]
    else:
        inference_period = [
            np.datetime64(x) for x in config["Global"]["inference_period"]
        ]
        low = inference_period[0]
        high = inference_period[1]
    return (low, high)


def check_date_format(df):
    df = df.reset_index()

    if (
        not pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
        .notnull()
        .all()
    ):
        raise ValueError("Invalid date format")

    return True


def check_date_bounds(df, date_low, date_high):
    if not ((date_low <= df["date"]) & (df["date"] < date_high)).all():
        raise ValueError("Date out of bounds")
    return True


def check_lad19cd_format(df):
    df = df.reset_index()

    # Must contain 9 characters, 1 region letter followed by 8 numbers
    split_code = df["lad19cd"].apply(lambda x: re.split("(\d+)", x))
    if not split_code.apply(
        lambda x: (len(x[0]) == 1) & (x[0] in "ENSW") & (len(x[1]) == 8)
    ).all():
        raise ValueError("Invalid lad19cd format")

    return True


def invalidInput(input):
    raise NotImplementedError(f'Input type "{input}" mode not implemented')
