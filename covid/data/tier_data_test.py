"""Tests Tier Data"""

import numpy as np
from covid.data import TierData


def test_url_tier_data():

    config = {
        "AreaCodeData": {
            "input": "json",
            "address": "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/LAD_APR_2019_UK_NC/FeatureServer/0/query?where=1%3D1&outFields=LAD19CD,LAD19NM&returnGeometry=false&returnDistinctValues=true&orderByFields=LAD19CD&outSR=4326&f=json",
            "format": "ons",
            "output": "processed_data/processed_lad19cd.csv",
            "regions": ["E"],
        },
        "TierData": {
            "input": "api",
            "address": None,
            "format": "api",
        },
        "GenerateOutput": {
            "storeInputs": True,
            "scrapedDataDir": "scraped_data",
            "storeProcessedInputs": True,
        },
        "Global": {
            "prependID": False,
            "prependDate": False,
            "inference_period": ["2020-10-12", "2021-01-04"],
        },
    }

    xarr = TierData.process(config)
    print("xarr", xarr)
    np.testing.assert_array_equal(xarr.shape, [315, 84, 6])
