"""Tests area codes"""

import pytest


def test_url():
    from covid.data import AreaCodeData

    config = {
        "AreaCodeData": {
            "input": "json",
            "address": "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/LAD_APR_2019_UK_NC/FeatureServer/0/query?where=1%3D1&outFields=LAD19CD,FID&returnGeometry=false&returnDistinctValues=true&orderByFields=LAD19CD&outSR=4326&f=json",
            "format": "ons",
            "output": "processed_data/processed_lad19cd.csv",
            "regions": ["E"],
        },
        "GenerateOutput": {
            "storeInputs": True,
            "scrapedDataDir": "scraped_data",
            "storeProcessedInputs": True,
        },
        "Global": {"prependID": False, "prependDate": False},
    }

    df = AreaCodeData.process(config)

    print(df)
