"""Methods to read in COVID-19 data and output
well-known formats"""

from warnings import warn
import numpy as np
import pandas as pd

__all__ = ["read_mobility", "read_population", "read_traffic_flow", "read_phe_cases"]


def read_mobility(path):
    """Reads in CSV with mobility matrix.

    CSV format: <To>,<id>,<id>,....
                <id>,<val>,<val>,...
                 ...

    :returns: a numpy matrix sorted by <id> on both rows and cols.
    """
    mobility = pd.read_csv(path)
    mobility = mobility[
        mobility["From"].str.startswith("E") & mobility["To"].str.startswith("E")
    ]
    mobility = mobility.sort_values(["From", "To"])
    mobility = mobility.groupby(["From", "To"]).agg({"Flow": sum}).reset_index()
    mob_matrix = mobility.pivot(index="To", columns="From", values="Flow")
    mob_matrix[mob_matrix.isna()] = 0.0
    return mob_matrix


def read_population(path):
    """Reads population CSV
    :returns: a pandas Series indexed by LTLAs
    """
    pop = pd.read_csv(path, index_col="lad19cd")
    pop = pop[pop.index.str.startswith("E")]
    pop = pop.sum(axis=1)
    pop = pop.sort_index()
    pop.name = "n"
    return pop


def read_traffic_flow(path: str, date_low: np.datetime64, date_high: np.datetime64):
    """Read traffic flow data, returning a timeseries between dates.
    :param path: path to a traffic flow CSV with <date>,<Car> columns
    :returns: a Pandas timeseries
    """
    commute_raw = pd.read_excel(
        path, index_col="Date", skiprows=5, usecols=["Date", "Cars"]
    )
    commute_raw.index = pd.to_datetime(commute_raw.index, format="%Y-%m-%d")
    commute_raw.sort_index(axis=0, inplace=True)
    commute = pd.DataFrame(index=np.arange(date_low, date_high, np.timedelta64(1, "D")))
    commute = commute.merge(commute_raw, left_index=True, right_index=True, how="left")
    commute[commute.index < commute_raw.index[0]] = commute_raw.iloc[0, 0]
    commute[commute.index > commute_raw.index[-1]] = commute_raw.iloc[-1, 0]
    commute["Cars"] = commute["Cars"] / 100.0
    commute.columns = ["percent"]
    return commute


def _merge_ltla(series):
    london = ["E09000001", "E09000033"]
    corn_scilly = ["E06000052", "E06000053"]
    series.loc[series.isin(london)] = ",".join(london)
    series.loc[series.isin(corn_scilly)] = ",".join(corn_scilly)
    return series


def read_phe_cases(
    path, date_low, date_high, pillar=None, date_type="specimen", ltlas=None
):
    """Reads a PHE Anonymised Line Listing for dates in [low_date, high_date)
    :param path: path to PHE Anonymised Line Listing Data
    :param low_date: lower date bound
    :param high_date: upper date bound
    :returns: a Pandas data frame of LTLAs x dates
    """
    date_type_map = {"specimen": "specimen_date", "report": "lab_report_date"}
    line_listing = pd.read_csv(
        path, usecols=[date_type_map[date_type], "LTLA_code", "pillar"]
    )[[date_type_map[date_type], "LTLA_code", "pillar"]]
    line_listing.columns = ["date", "lad19cd", "pillar"]
    line_listing["lad19cd"] = _merge_ltla(line_listing["lad19cd"])
    line_listing["date"] = pd.to_datetime(line_listing["date"], format="%d/%m/%Y")
    # Select dates
    line_listing = line_listing[
        (date_low <= line_listing["date"])
        & (line_listing["date"] < (date_high - np.timedelta64(1, "D")))
    ]

    # Choose pillar
    if pillar is not None:
        line_listing = line_listing.loc[line_listing["pillar"] == pillar]

    # Drop na rows
    orig_len = line_listing.shape[0]
    line_listing = line_listing.dropna(axis=0)
    warn(
        f"Removed {orig_len - line_listing.shape[0]} rows of {orig_len} \
due to missing values ({100. * (orig_len - line_listing.shape[0])/orig_len}%)"
    )

    # Aggregate by date/region
    case_counts = line_listing.groupby(["date", "lad19cd"]).size()
    case_counts.name = "count"

    # Re-index
    dates = pd.date_range(date_low, date_high, closed="left")
    if ltlas is None:
        ltlas = case_counts.index.levels[1]
    index = pd.MultiIndex.from_product([dates, ltlas], names=["date", "lad19cd"])
    case_counts = case_counts.reindex(index, fill_value=0)
    return case_counts.reset_index().pivot(
        index="lad19cd", columns="date", values="count"
    )
