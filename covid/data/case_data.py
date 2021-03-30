"""Loads COVID-19 case data"""

import time
from warnings import warn
import requests
import json
import numpy as np
import pandas as pd

from covid.data.util import (
    invalidInput,
    get_date_low_high,
    check_date_bounds,
    check_date_format,
    check_lad19cd_format,
    merge_lad_codes,
)
from covid.data import AreaCodeData


class CasesData:
    def get(config):
        """
        Retrieve a pandas DataFrame containing the cases/line list data.
        """
        settings = config["CasesData"]
        if settings["input"] == "url":
            df = CasesData.getURL(settings["address"], config)
        elif settings["input"] == "csv":
            print(
                "Reading case data from local CSV file at", settings["address"]
            )
            df = CasesData.getCSV(settings["address"])
        elif settings["input"] == "processed":
            print(
                "Reading case data from preprocessed CSV at",
                settings["address"],
            )
            df = pd.read_csv(settings["address"], index_col=0)
        else:
            invalidInput(settings["input"])

        return df

    def getURL(url, config):
        """
        Placeholder, in case we wish to interface with an API.
        """
        max_tries = 5
        secs = 5
        for i in range(max_tries):
            try:
                print("Attempting to download...", end="", flush=True)
                response = requests.get(url)
                content = json.loads(response.content)
                df = pd.read_json(json.dumps(content["body"]))
                print("Success", flush=True)
                return df
            except (requests.ConnectionError, requests.RequestException) as e:
                print("Failed", flush=True)
                print(e)
                time.sleep(secs * 2 ** i)

        raise ConnectionError(
            f"Data download timed out after {max_tries} attempts"
        )

    def getCSV(file):
        """
        Format as per linelisting
        """
        columns = ["pillar", "LTLA_code", "specimen_date", "lab_report_date"]
        dfs = pd.read_csv(file, chunksize=50000, iterator=True, usecols=columns)
        df = pd.concat(dfs)
        return df

    def check(df, config):
        """
        Check that data format seems correct
        """
        nareas = len(config["lad19cds"])
        date_low, date_high = get_date_low_high(config)
        dates = pd.date_range(start=date_low, end=date_high, closed="left")
        days = len(dates)
        entries = days * nareas

        if not (
            ((dims[1] >= 3) & (dims[0] == entries))
            | ((dims[1] == days) & (dims[0] == nareas))
        ):
            print(df)
            raise ValueError("Incorrect CasesData dimensions")

        if "date" in df:
            _df = df
        elif df.columns.name == "date":
            _df = pd.DataFrame({"date": df.columns})
        else:
            raise ValueError("Cannot determine date axis")

        check_date_bounds(df, date_low, date_high)
        check_date_format(df)
        check_lad19cd_format(df)
        df = df.rename(columns={"date": "time"})
        return True

    def adapt(df, config):
        """
        Adapt the line listing data to the desired dataframe format.
        """
        # Extract the yaml config settings
        date_low, date_high = get_date_low_high(config)
        settings = config["CasesData"]
        pillars = settings["pillars"]
        measure = settings["measure"].casefold()

        # this key might not be stored in the config file
        # if it's not, we need to grab it using AreaCodeData
        if "lad19cds" not in config:
            _df = AreaCodeData.process(config)
        areacodes = config["lad19cds"]

        if settings["input"] == "processed":
            return df

        if settings["format"].lower() == "phe":
            df = CasesData.adapt_phe(
                df,
                date_low,
                date_high,
                pillars,
                measure,
                areacodes,
            )
        elif (settings["input"] == "url") and (settings["format"] == "json"):
            df = CasesData.adapt_gov_api(
                df, date_low, date_high, pillars, measure, areacodes
            )

        return df

    def adapt_gov_api(df, date_low, date_high, pillars, measure, areacodes):

        warn("Using API data: 'pillar' and 'measure' will be ignored")

        df = df.rename(
            columns={"areaCode": "location", "newCasesBySpecimenDate": "cases"}
        )
        df = df[["location", "date", "cases"]]
        df["date"] = pd.to_datetime(df["date"])
        df["location"] = merge_lad_codes(df["location"])
        df = df[df["location"].isin(areacodes)]
        df.index = pd.MultiIndex.from_frame(df[["location", "date"]])
        df = df.sort_index()

        dates = pd.date_range(date_low, date_high, closed="left")
        multi_index = pd.MultiIndex.from_product([areacodes, dates])
        ser = df["cases"].reindex(multi_index, fill_value=0.0)
        ser.index.names = ["location", "time"]
        ser.name = "cases"
        return ser

    def adapt_phe(df, date_low, date_high, pillars, measure, areacodes):
        """
        Adapt the line listing data to the desired dataframe format.
        """
        # Clean missing values
        df.dropna(inplace=True)
        df = df.rename(columns={"LTLA_code": "lad19cd"})

        # Clean time formats
        df["specimen_date"] = pd.to_datetime(df["specimen_date"], dayfirst=True)
        df["lab_report_date"] = pd.to_datetime(
            df["lab_report_date"], dayfirst=True
        )

        df["lad19cd"] = merge_lad_codes(df["lad19cd"])

        # filters for pillars, date ranges, and areacodes if given
        filters = df["pillar"].isin(pillars)
        filters &= df["lad19cd"].isin(areacodes)
        if measure == "specimen":
            filters &= (date_low <= df["specimen_date"]) & (
                df["specimen_date"] < date_high
            )
        else:
            filters &= (date_low <= df["lab_report_date"]) & (
                df["lab_report_date"] < date_high
            )
        df = df[filters]
        df = df.drop(columns="pillar")  # No longer need pillar column

        # Aggregate counts
        if measure == "specimen":
            df = df.groupby(["lad19cd", "specimen_date"]).count()
            df = df.rename(columns={"lab_report_date": "cases"})
        else:
            df = df.groupby(["lad19cd", "lab_report_date"]).count()
            df = df.rename(columns={"specimen_date": "cases"})

        df.index.names = ["lad19cd", "time"]
        df = df.sort_index()

        # Fill in all dates, and add 0s for empty counts
        dates = pd.date_range(date_low, date_high, closed="left")
        multi_indexes = pd.MultiIndex.from_product(
            [areacodes, dates], names=["location", "time"]
        )
        results = df["cases"].reindex(multi_indexes, fill_value=0.0)
        return results.sort_index()

    def process(config):
        df = CasesData.get(config)
        df = CasesData.adapt(df, config)
        return df
