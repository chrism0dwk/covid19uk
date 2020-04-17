import unittest

from covid.pydata import phe_linelist_timeseries, zero_cases, collapse_pop


class TestHiddenEventInitialize(unittest.TestCase):

    def setUp(self) -> None:
        linelist = phe_linelist_timeseries('../data/DailyConfirmedCases_2020-04-04.csv')
        pop = collapse_pop('../data/c2019modagepop.csv')
        self.cases = zero_cases(linelist, pop)

    def test_one_event_init(self):
        from covid.util import initialise_previous_events
        cases = self.cases.loc['2020-02-27']
        events = initialise_previous_events(cases, 0.5)
        self.assertEqual(cases.sum(), events.sum())

    def test_all_event_init(self):
        from covid.util import initialise_previous_events
        events = self.cases.groupby(level=0, axis=0).apply(
            lambda cases: initialise_previous_events(cases, 0.5))
        events = events.sum(level=list(range(1, 4)))
        self.assertEqual(self.cases.sum(), events.sum())


if __name__ == '__main__':
    unittest.main()
