"""Provides functions to format data"""

import pandas as pd


def _expand_quantiles(q_dict):
    """Expand a dictionary of quantiles"""
    q_str = [
        "0.05",
        "0.1",
        "0.15",
        "0.2",
        "0.25",
        "0.3",
        "0.35",
        "0.4",
        "0.45",
        "0.5",
        "0.55",
        "0.6",
        "0.65",
        "0.7",
        "0.75",
        "0.8",
        "0.85",
        "0.9",
        "0.95",
    ]
    quantiles = {f"Quantile {q}": None for q in q_str}
    if q_dict is None:
        return quantiles

    for k, v in q_dict.items():
        q_key = f"Quantile {k}"
        if q_key not in quantiles.keys():
            raise KeyError(f"quantile '{k}' not compatible with template form")
        quantiles[q_key] = v
    return [
        pd.Series(v, name=k).reset_index(drop=True)
        for k, v in quantiles.items()
    ]


def _split_dates(dates):
    if dates is None:
        return {"day": None, "month": None, "year": None}
    if hasattr(dates, "__iter__"):
        dx = pd.DatetimeIndex(dates)
    else:
        dx = pd.DatetimeIndex([dates])
    return {"day": dx.day, "month": dx.month, "year": dx.year}


def make_dstl_template(
    group=None,
    model=None,
    scenario=None,
    model_type=None,
    version=None,
    creation_date=None,
    value_date=None,
    age_band=None,
    geography=None,
    value_type=None,
    value=None,
    quantiles=None,
):
    """Formats a DSTL-type Excel results template"""

    # Process date
    creation_date_parts = _split_dates(creation_date)
    value_date_parts = _split_dates(value_date)
    quantile_series = _expand_quantiles(quantiles)

    fields = {
        "Group": group,
        "Model": model,
        "Scenario": scenario,
        "ModelType": model_type,
        "Version": version,
        "Creation Day": creation_date_parts["day"],
        "Creation Month": creation_date_parts["month"],
        "Creation Year": creation_date_parts["year"],
        "Day of Value": value_date_parts["day"],
        "Month of Value": value_date_parts["month"],
        "Year of Value": value_date_parts["year"],
        "AgeBand": age_band,
        "Geography": geography,
        "ValueType": value_type,
        "Value": value,
    }
    fields = [
        pd.Series(v, name=k).reset_index(drop=True) for k, v in fields.items()
    ]
    return pd.concat(fields + quantile_series, axis="columns").ffill(
        axis="index"
    )
