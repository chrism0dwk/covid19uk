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


if __name__=='__main__':
    pass
