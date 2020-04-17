"""Python-based data munging"""

import os
from warnings import warn

import geopandas as gp
import numpy as np
import pandas as pd
import pyreadr as pyr


def load_commute_volume(filename, date_range):
    """Loads commute data and clips or extends date range"""
    commute_raw = pd.read_csv(filename, index_col='date')
    commute_raw.index = pd.to_datetime(commute_raw.index, format='%Y-%m-%d')
    commute_raw.sort_index(axis=0, inplace=True)
    commute = pd.DataFrame(index=np.arange(date_range[0], date_range[1], np.timedelta64(1,'D')))
    commute = commute.merge(commute_raw, left_index=True, right_index=True, how='left')
    commute[commute.index < commute_raw.index[0]] = commute_raw.iloc[0, 0]
    commute[commute.index > commute_raw.index[-1]] = commute_raw.iloc[-1, 0]
    return commute


def group_ages(df):
    """
    Sums age groups
    :param df: a dataframe with columns 0,1,2,...,90
    :return: a dataframe with 5-year age groups
    """
    ages = np.arange(90).reshape([90//5, 5]).astype(np.str)
    grouped_ages = pd.DataFrame()
    for age_group in ages:
        grouped_ages[f"[{age_group[0]}-{int(age_group[-1])+1})"] = df[age_group].sum(axis=1)
    grouped_ages['[90,)'] = df[['90']]
    grouped_ages['[80,inf)'] = grouped_ages[['[80-85)', '[85-90)', '[90,)']].sum(axis=1)
    grouped_ages = grouped_ages.drop(columns=['[80-85)', '[85-90)', '[90,)'])
    return grouped_ages


def ingest_data(lad_shp, lad_pop):
    pop = pd.read_csv(lad_pop, skiprows=4, thousands=',')
    age_pop = group_ages(pop)
    age_pop.index = pop['Code']

    lad = gp.read_file(lad_shp)
    lad.index = lad['lad19cd'].rename('Code')
    lad = lad.iloc[lad.index.str.match('^E0[6-9]'), :]
    lad = lad.merge(age_pop, on='Code')
    lad.sort_index(inplace=True)
    lad.drop(columns=['objectid', 'lad19cd' ,'long', 'lat'])

    N = lad.iloc[:, lad.columns.str.match(pat='^[[0-9]')].stack()

    print(f"Found {lad.shape[0]} LADs")

    return {'geo': lad, 'N': N}


def phe_death_timeseries(filename, date_range=['2020-02-02', '2020-03-21']):
    date_range = [np.datetime64(x) for x in date_range]
    csv = pd.read_excel(filename)
    cases = pd.DataFrame({'hospital': csv.groupby(['Hospital admission date (non-HCID)', 'Region']).size(),
                          'deaths': csv.groupby(['PATIENT_DEATH_DATE', 'Region']).size()})
    cases.index.rename(['date', 'region'], [0, 1], inplace=True)
    cases.reset_index(inplace=True)
    cases = cases.pivot(index='date', columns='region')
    dates = pd.DataFrame(index=pd.DatetimeIndex(np.arange(*date_range, np.timedelta64(1, 'D'))))
    combined = dates.merge(cases, how='left', left_index=True, right_index=True)
    combined.columns = pd.MultiIndex.from_tuples(combined.columns, names=['timeseries','region'])
    combined[combined.isna()] = 0.0

    output = {k: combined.loc[:, [k, None]] for k in combined.columns.levels[0]}
    return output


def phe_death_hosp_to_death(filename, date_range=['2020-02-02', '2020-03-21']):
    date_range = [np.datetime64(x) for x in date_range]
    csv = pd.read_excel(filename)

    data = csv.loc[:, ['Sex', 'Age', 'Underlying medical condition?', 'Hospital admission date (non-HCID)',
                   'PATIENT_DEATH_DATE']]
    data.columns = ['sex','age','underlying_condition', 'hosp_adm_date', 'death_date']
    data.loc[:, 'underlying_condition'] = data['underlying_condition'] == 'Yes'
    data['adm_to_death'] = (data['death_date'] - data['hosp_adm_date']) / np.timedelta64(1, 'D')
    return data.dropna(axis=0)


def phe_linelist_timeseries(filename, spec_date='specimen_date', utla='UTLA_code', age='Age',
                            date_range=None):

    linelist = pd.read_csv(filename)
    linelist = linelist[[spec_date, utla, age]]

    # 1. clip dates
    one_day = np.timedelta64(1, 'D')
    linelist[spec_date] = pd.to_datetime(linelist[spec_date], format="%d/%m/%Y")
    date_range = date_range or [linelist[spec_date].min(), linelist[spec_date].max()]
    linelist = linelist[(date_range[0] <= linelist[spec_date]) & (linelist[spec_date] <= date_range[1])]
    raw_len = linelist.shape[0]

    # 2. Remove NA rows
    linelist = linelist.dropna(axis=0)  # remove na's
    warn(f"Removed {raw_len - linelist.shape[0]} rows of {raw_len} due to missing data \
({100. * (raw_len - linelist.shape[0])/raw_len}%)")

    # 2a. Aggregate London/Westminster and Cornwall/Scilly
    london = ['E09000001', 'E09000033']
    corn_scilly = ['E06000052', 'E06000053']
    linelist.loc[linelist[utla].isin(london), utla] = ','.join(london)
    linelist.loc[linelist[utla].isin(corn_scilly), utla] = ','.join(corn_scilly)

    # 3. Create age groups
    linelist['age_group'] = np.clip(linelist[age] // 5, a_min=0, a_max=16).astype(np.int64) * 5  # id of 5-year age group

    # 4. Group by UTLA/age
    case_counts = linelist.groupby([spec_date, utla, 'age_group']).size()
    case_counts.sort_index(axis=0, inplace=True)

    return case_counts


def zero_cases(case_timeseries, population):
    """Creates a full case matrix, filling in dates, lads, and age groups not represented
    in the main dataset.  It is explicitly assumed that missing date/lad/age combos in the
    case_timeseries are true 0s.
    :param case_timeseries: an indexed [date, UTLA_code, age_group] pd.Series containing case counts
    :param population: a dataset indexed with all UTLA_codes and age_groups
    """
    dates = np.arange(case_timeseries.index.levels[0].min(),
                      case_timeseries.index.levels[0].max() + np.timedelta64(1, 'D'), # inclusive interval
                      np.timedelta64(1, 'D'))
    fullidx = pd.MultiIndex.from_product([dates, *population.index.levels])
    y = case_timeseries.reindex(fullidx)
    y[y.isna()] = 0. # Big assumption that a missing value is a true 0!
    return y


def collapse_commute_data(flow_file):
    """Collapses LTLA-based commuting data in England to UTLA areas.

    Merges commuting data at LTLA areal basis onto modified UTLA Dec 2019 area.

    Modifications:
    E06000052, E06000053 combined
    E09000001, E09000033 combined
    """
    filedir = os.path.dirname(os.path.abspath(__file__))
    commuting = list(pyr.read_r(flow_file).values())[0]
    lt_map = pd.read_csv(filedir + '/../data/Lower_Tier_Local_Authority_to_Upper_Tier_Local_Authority_April_2019_Lookup_in_England_and_Wales.csv')
    lt_map = lt_map[['LTLA19CD', 'UTLA19CD']]

    # 1. Extract England
    commuting = commuting[commuting['From'].str.startswith('E') & commuting['To'].str.startswith('E')]

    # 1. Merge in lt_map on 'From' field
    def merge(left, right, left_on, right_on, new_cols):
        merged = left.merge(right, how='left', left_on=left_on, right_on=right_on)
        colnames = merged.columns.to_numpy()
        colnames[-len(new_cols):] = new_cols
        merged.columns = pd.Index(colnames)
        return merged

    commuting = merge(commuting, lt_map, 'From', 'LTLA19CD', ['from_ltla', 'from_utla'])
    commuting = merge(commuting, lt_map, 'To', 'LTLA19CD', ['to_ltla', 'to_utla'])

    # 2. Fix up collapsed UTLAs
    commuting.loc[commuting['From'].str.contains(','), 'from_utla'] = commuting.loc[
        commuting['From'].str.contains(','), 'From']
    commuting.loc[commuting['To'].str.contains(','), 'to_utla'] = commuting.loc[
        commuting['To'].str.contains(','), 'To']

    # 3. Collapse data
    collapsed = commuting.groupby(['from_utla', 'to_utla']).agg({'Flow': sum})
    collapsed.sort_index(inplace=True)
    collapsed.reset_index(inplace=True)

    # 4. Pivot to return a matrix
    commute_matrix = collapsed.pivot(index='to_utla', columns='from_utla', values='Flow')
    commute_matrix[commute_matrix.isna()] = 0.0

    return commute_matrix


def collapse_pop(pop_file):
    """Aggregates LTLA2019 population data to UTLA2019 and 5-year age groups to 80+"""
    filedir = os.path.dirname(os.path.abspath(__file__))
    pop = pd.read_csv(pop_file)
    pop = pop[pop['lad19cd'].str.startswith('E')]

    lt_map = pd.read_csv(filedir + '/../data/Lower_Tier_Local_Authority_to_Upper_Tier_Local_Authority_April_2019_Lookup_in_England_and_Wales.csv')

    # 1. Merge LADs
    pop = pop.merge(lt_map['UTLA19CD'], how='left', left_on='lad19cd', right_on=lt_map['LTLA19CD'])

    # 2. Fill in merged utla codes
    pop.loc[pop['lad19cd'].str.contains(','), 'UTLA19CD'] = pop.loc[pop['lad19cd'].str.contains(','), 'lad19cd']
    pop.index = pd.MultiIndex.from_frame(pop[['UTLA19CD', 'lad19cd']])
    pop.drop(columns=['lad19cd', 'UTLA19CD'], inplace=True)
    pop.columns = np.arange(pop.shape[1]) * 5  # 5 year age groups
    pop.sort_index(inplace=True)

    # 3. Aggregate by UTLA19CD
    pop = pop.sum(level=0)
    pop.iloc[:, -3] = pop.iloc[:, -3:].sum(axis=1)
    pop = pop.iloc[:, :-2]

    # 4. Long format
    pop = pop.reset_index().melt(id_vars=['UTLA19CD'], value_name='n', var_name='age_group')
    pop.index = pd.MultiIndex.from_frame(pop[['UTLA19CD', 'age_group']])
    pop.drop(columns=['UTLA19CD', 'age_group'], inplace=True)
    pop.sort_index(level=0, inplace=True)

    return pop

if __name__=='__main__':

    ts = phe_linelist_timeseries('/home/jewellcp/Insync/jewellcp@lancaster.ac.uk/OneDrive Biz - Shared/covid19/data/PHE_2020-04-01/Anonymised Line List 20200401.csv')
    print(ts)
