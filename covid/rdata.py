"""Loads R data.frame data structures"""

import numpy
import pyreadr as pyr
import numpy as np


def load_age_mixing(rds_file: str):
    """Loads age mixing matrix from R.

    :param rds_file: a .rds file containing an R data.frame with mixing matrix
    """
    raw = pyr.read_r(rds_file)
    K = list(raw.values())[0]
    age_groups = K.columns
    return K.to_numpy(dtype=np.float32), age_groups


def load_mobility_matrix(rds_file: str):
    """Loads mobility COO from RDS file.

     :param rds_file: a .rds file containing an R data.frame with columns 'Residence',
     and 'Workplace' indicating matrix coordinates, and first column containing the value
     """
    raw = pyr.read_r(rds_file)
    df = list(raw.values())[0]
    colnames = df.columns
    mobility_matrix = df.pivot(index='Workplace', columns='Residence', values=colnames[0])
    mobility_matrix[mobility_matrix.isna()] = 0.
    return mobility_matrix.to_numpy(dtype=np.float32), mobility_matrix.index.to_numpy()


def load_population(rds_file: str):
    """Loads population data from RDS file.

    :param rds_file: and RDS file containing a data.frame with columns 'age', 'LA.code', 'n'
    """
    raw = pyr.read_r(rds_file)
    df = list(raw.values())[0]
    df = df.sort_values(by=['LA.code', 'age'])
    return df[['LA.code', 'name', 'Area.name.2', 'age', 'n']]
