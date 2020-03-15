import unittest
import matplotlib.pyplot as plt


class TestData(unittest.TestCase):

    def test_ingest_data(self):
        from covid.pydata import ingest_data
        x = ingest_data('../data/lad_shp/Local_Authority_Districts_December_2019_Boundaries_UK_BFC.shp',
                        '../data/ukmidyearestimates20182019ladcodes.csv')
        x['geo'].plot(column='[80,inf)', legend=True)
        plt.show()


if __name__ == '__main__':
    unittest.main()
