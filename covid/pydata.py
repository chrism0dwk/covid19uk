"""Python-based data munging"""

import numpy as np
import pandas as pd
import geopandas as gp

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





if __name__=='__main__':
    pass
