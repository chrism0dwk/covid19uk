import unittest
import numpy as np
import matplotlib.pyplot as plt


class TestData(unittest.TestCase):

    def test_ingest_data(self):
        from covid.pydata import ingest_data
        x = ingest_data('../data/lad_shp/Local_Authority_Districts_December_2019_Boundaries_UK_BFC.shp',
                        '../data/ukmidyearestimates20182019ladcodes.csv')
        x['geo'].plot(column='[80,inf)', legend=True)
        plt.show()

    @unittest.skip
    def test_phe_linelisting_timeseries(self):
        from covid.pydata import phe_linelist_timeseries
        ts = phe_linelist_timeseries('/home/jewellcp/Insync/jewellcp@lancaster.ac.uk/OneDrive Biz - Shared/covid19/data/PHE_2020-04-01/Anonymised Line List 20200401.csv')
        np.testing.assert_array_equal(ts.index.levels[2].isin(range(0, 17)), [True]*17)


if __name__ == '__main__':
    unittest.main()
