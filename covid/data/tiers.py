"""Import COVID Tier data"""

import numpy as np
import pandas as pd

from covid.data.area_code import AreaCodeData
from covid.data.util import get_date_low_high, invalidInput, merge_lad_codes


class TierData:
    def get(config):
        """
        Retrieve an xarray DataArray of the tier data
        """
        settings = config["TierData"]
        if settings["input"] == "csv":
            df = TierData.getCSV(settings["address"])
        elif settings["input"] == "api":
            df = TierData.getCSV(
                "https://api.coronavirus.data.gov.uk/v2/data?areaType=ltla&metric=alertLevel&format=csv"
            )
        else:
            invalidInput(settings["input"])

        return df

    def getCSV(file):
        """
        Read TierData CSV from file
        """
        return pd.read_csv(file)

    def check(xarray, config):
        """
        Check the data format
        """
        return True

    def adapt(df, config):
        """
        Adapt the dataframe to the desired format.
        """
        settings = config["TierData"]

        # TODO this key might not be stored in the config file
        # if it's not, we need to grab it using AreaCodeData
        if "lad19cds" not in config:
            areacodes = AreaCodeData.process(config)["lad19cd"]
        else:
            areacodes = config["lad19cds"]

        # Below is assuming inference_period dates
        date_low, date_high = get_date_low_high(config)

        if settings["format"].lower() == "tidy":
            xarray = TierData.adapt_xarray(
                df, date_low, date_high, areacodes, settings
            )
        elif settings["format"].lower() == "api":
            xarray = TierData.adapt_api_xarray(
                df, date_low, date_high, areacodes, settings
            )

        return xarray

    def adapt_api_xarray(tiers, date_low, date_high, lads, settings):
        """
        Adapt web-api to desired format
        """
        tiers["date"] = pd.to_datetime(tiers["date"], format="%Y-%m-%d")
        tiers["lad19cd"] = merge_lad_codes(tiers["areaCode"])
        tiers["alert_level"] = tiers["alertLevel"]
        tiers = tiers[["date", "lad19cd", "alert_level"]]

        if len(lads) > 0:
            tiers = tiers[tiers["lad19cd"].isin(lads)]

        date_range = pd.date_range(date_low, date_high - np.timedelta64(1, "D"))

        def interpolate(df):
            df.index = pd.Index(pd.to_datetime(df["date"]), name="date")
            df = df.drop(columns="date").sort_index()
            df = df.reindex(date_range)
            df["alert_level"] = (
                df["alert_level"].ffill().backfill().astype("int")
            )
            return df[["alert_level"]]

        tiers = tiers.groupby(["lad19cd"]).apply(interpolate)
        tiers = tiers.reset_index()
        tiers.columns = ["lad19cd", "date", "alert_level"]

        index = pd.MultiIndex.from_frame(tiers)
        index = index.sort_values()
        index = index[~index.duplicated()]
        ser = pd.Series(1, index=index, name="value")
        ser = ser.loc[
            pd.IndexSlice[:, date_low : (date_high - np.timedelta64(1, "D")), :]
        ]
        xarr = ser.to_xarray()
        xarr.data[np.isnan(xarr.data)] = 0.0
        # return [T, M, V] structure
        return xarr.transpose("date", "lad19cd", "alert_level")

    def adapt_xarray(tiers, date_low, date_high, lads, settings):
        """
        Adapt to a filtered xarray object
        """
        tiers["date"] = pd.to_datetime(tiers["date"], format="%Y-%m-%d")
        tiers["code"] = merge_lad_codes(tiers["code"])

        # Separate out December tiers
        date_mask = tiers["date"] > np.datetime64("2020-12-02")
        tiers.loc[
            date_mask & (tiers["tier"] == "three"),
            "tier",
        ] = "dec_three"
        tiers.loc[
            date_mask & (tiers["tier"] == "two"),
            "tier",
        ] = "dec_two"
        tiers.loc[
            date_mask & (tiers["tier"] == "one"),
            "tier",
        ] = "dec_one"

        # filter down to the lads
        if len(lads) > 0:
            tiers = tiers[tiers.code.isin(lads)]

        # add in fake LADs to ensure all lockdown tiers are present for filtering
        # xarray.loc does not like it when the values aren't present
        # this seems to be the cleanest way
        # we drop TESTLAD after filtering down
        # lockdown_states = ["two", "three", "dec_two", "dec_three"]
        lockdown_states = settings["lockdown_states"]

        for (i, t) in enumerate(lockdown_states):
            tiers.loc[tiers.shape[0] + i + 1] = [
                "TESTLAD",
                "TEST",
                "LAD",
                date_low,
                t,
            ]

        index = pd.MultiIndex.from_frame(tiers[["date", "code", "tier"]])
        index = index.sort_values()
        index = index[~index.duplicated()]
        ser = pd.Series(1.0, index=index, name="value")
        ser = ser[date_low : (date_high - np.timedelta64(1, "D"))]
        xarr = ser.to_xarray()
        xarr.data[np.isnan(xarr.data)] = 0.0
        xarr_filt = xarr.loc[..., lockdown_states]
        xarr_filt = xarr_filt.drop_sel({"code": "TESTLAD"})
        return xarr_filt

    def process(config):
        if config["TierData"]["format"].lower()[0:5] == "lancs":
            xarray = TierData.process_lancs(config)
        else:
            df = TierData.get(config)
            xarray = TierData.adapt(df, config)
        if TierData.check(xarray, config):
            return xarray

    def process_lancs(config):
        global_settings = config["Global"]
        settings = config["TierData"]
        if "lad19cds" not in config:
            _df = AreaCodeData.process(config)
        areacodes = config["lad19cds"]
        date_low, date_high = get_date_low_high(config)
        if config["TierData"]["format"].lower() == "lancs_raw":
            return LancsData.read_tier_restriction_data(
                settings["address"], areacodes, date_low, date_high
            )
        elif config["TierData"]["format"].lower() == "lancs_tidy":
            return LancsData.read_challen_tier_restriction(
                settings["address"], date_low, date_high, areacodes
            )
        elif config["TierData"]["format"].lower() == "api":
            raise NotImplementedError(f"Tier data api not implemented")
        else:
            raise NotImplementedError(
                f'Format type {config["TierData"]["format"]} not implemented'
            )
