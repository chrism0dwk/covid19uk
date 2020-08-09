"""Tests for covid.util functions"""

import unittest
import os
import numpy as np


from covid.util import regularize_occults

# pylint: disable=missing-class-docstring,missing-function-docstring


class TestRegularizeOccults(unittest.TestCase):
    def setUp(self):
        thisdir = os.path.dirname(__file__)
        self.fixture = os.path.join(thisdir, "fixture/test_regularize_occults.npz")

    def test_regularize_occults(self):
        data = np.load(self.fixture)
        stoichiometry = np.array(
            [[-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1]], dtype=np.float64
        )

        state, occults = regularize_occults(
            data["events"], data["occults"], data["init_state"], stoichiometry
        )
        self.assertEqual(np.sum(state < 0), 0)
        self.assertEqual(np.sum(occults < 0), 0)


if __name__ == "__main__":
    unittest.main()
