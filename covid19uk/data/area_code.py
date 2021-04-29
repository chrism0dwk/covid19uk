"""Retrieves LAD19 area codes"""

from http import HTTPStatus
import json

import pandas as pd
import requests

from covid19uk.data.util import (
    merge_lad_codes,
    check_lad19cd_format,
    invalidInput,
)


class AreaCodeData:
    def get(config):
        """
        Retrieve a response containing a list of all the LAD codes
        """

        settings = config["AreaCodeData"]
        if settings["input"] == "url":
            df = AreaCodeData.getURL(settings["address"], config)
            df.columns = [x.lower() for x in df.columns]
        elif settings["input"] == "json":
            print(
                "Reading Area Code data from local JSON file at",
                settings["address"],
            )
            df = AreaCodeData.getJSON(settings["address"])
        elif settings["input"] == "csv":
            print(
                "Reading Area Code data from local CSV file at",
                settings["address"],
            )
            df = AreaCodeData.getCSV(settings["address"])
        elif settings["input"] == "processed":
            print(
                "Reading Area Code data from preprocessed CSV at",
                settings["address"],
            )
            df = pd.read_csv(settings["address"])
        else:
            invalidInput(settings["input"])

        return df

    def getConfig(config):
        # Create a dataframe from the LADs specified in config
        df = pd.DataFrame(config["lad19cds"], columns=["lad19cd"])
        df["name"] = "n/a"  # placeholder names for now.
        return df

    def getURL(url, config):
        settings = config["AreaCodeData"]

        fields = ["LAD19CD", "LAD19NM"]

        api_params = {"outFields": str.join(",", fields), "f": "json"}

        response = requests.get(url, params=api_params, timeout=5)
        if response.status_code >= HTTPStatus.BAD_REQUEST:
            raise RuntimeError(f"Request failed: {response.text}")

        if settings["format"] == "ons":
            print("Retrieving Area Code data from the ONS")
            data = response.json()
            df = AreaCodeData.getJSON(json.dumps(data))

        return df

    def cmlad11_to_lad19(cmlad11):
        """
        Converts CM (census merged) 2011 codes to LAD 2019 codes
        """
        # The below URL converts from CMLAD2011CD to LAD11CD
        #        url = "http://infuse.ukdataservice.ac.uk/showcase/mergedgeographies/Merging-Local-Authorities-Lookup.xlsx"
        #        response = requests.get(url, timeout=5)
        #        if response.status_code >= HTTPStatus.BAD_REQUEST:
        #            raise RuntimeError(f'Request failed: {response.text}')
        #
        #        data = io.BytesIO(response.content)
        #
        #        cm11_to_lad11_map = pd.read_excel(data)

        # cached
        cm11_to_lad11_map = pd.read_excel(
            "data/Merging-Local-Authorities-Lookup.xlsx"
        )

        cm11_to_lad11_dict = dict(
            zip(
                cm11_to_lad11_map["Merging Local Authority Code"],
                cm11_to_lad11_map["Standard Local Authority Code"],
            )
        )

        lad19cds = cmlad11.apply(
            lambda x: cm11_to_lad11_dict[x]
            if x in cm11_to_lad11_dict.keys()
            else x
        )

        mapping = {
            "E06000028": "E06000058",  # "Bournemouth" : "Bournemouth, Christchurch and Poole",
            "E06000029": "E06000058",  # "Poole" : "Bournemouth, Christchurch and Poole",
            "E07000048": "E06000058",  # "Christchurch" : "Bournemouth, Christchurch and Poole",
            "E07000050": "E06000059",  # "North Dorset" : "Dorset",
            "E07000049": "E06000059",  # "East Dorset" : "Dorset",
            "E07000052": "E06000059",  # "West Dorset" : "Dorset",
            "E07000051": "E06000059",  # "Purbeck" : "Dorset",
            "E07000053": "E06000059",  # "Weymouth and Portland" : "Dorset",
            "E07000191": "E07000246",  # "West Somerset" : "Somerset West and Taunton",
            "E07000190": "E07000246",  # "Taunton Deane" : "Somerset West and Taunton",
            "E07000205": "E07000244",  # "Suffolk Coastal" : "East Suffolk",
            "E07000206": "E07000244",  # "Waveney" : "East Suffolk",
            "E07000204": "E07000245",  # "St Edmundsbury" : "West Suffolk",
            "E07000201": "E07000245",  # "Forest Heath" : "West Suffolk",
            "E07000097": "E07000242",  # East Hertforshire
            "E07000101": "E07000243",  # Stevenage
            "E07000100": "E07000240",  # St Albans
            "E08000020": "E08000037",  # Gateshead
            "E06000048": "E06000057",  # Northumberland
            "E07000104": "E07000241",  # Welwyn Hatfield
        }

        lad19cds = lad19cds.apply(
            lambda x: mapping[x] if x in mapping.keys() else x
        )
        lad19cds = merge_lad_codes(lad19cds)

        return lad19cds

    def getJSON(file):
        data = pd.read_json(file, orient="index").T["features"][0]
        data = [record["attributes"] for record in data]
        df = pd.DataFrame.from_records(data)
        return df

    def getCSV(file):
        return pd.read_csv(file)

    def check(df, config):
        """
        Check that data format seems correct
        """
        check_lad19cd_format(df)
        return True

    def adapt(df, config):
        """
        Adapt the area codes to the desired dataframe format
        """
        settings = config["AreaCodeData"]
        regions = settings["regions"]

        if settings["input"] == "processed":
            return df

        if settings["format"].lower() == "ons":
            df = AreaCodeData.adapt_ons(df, regions)

        # if we have a predefined list of LADs, filter them down
        if "lad19cds" in config:
            df = df[[x in config["lad19cds"] for x in df.lad19cd.values]]

        return df

    def adapt_ons(df, regions):
        colnames = ["lad19cd", "name"]
        df.columns = colnames
        filters = df["lad19cd"].str.contains(str.join("|", regions))
        df = df[filters]
        df["lad19cd"] = merge_lad_codes(df["lad19cd"])
        df = df.drop_duplicates(subset="lad19cd")

        return df

    def process(config):
        df = AreaCodeData.get(config)
        df = AreaCodeData.adapt(df, config)
        if AreaCodeData.check(df, config):
            config["lad19cds"] = df["lad19cd"].tolist()
            return df
