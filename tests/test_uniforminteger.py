import unittest
import numpy as np
import tensorflow as tf

from covid.impl.UniformInteger import UniformInteger


class TestUniformInteger(unittest.TestCase):
    def setUp(self):
        self.seed = 10402302
        self.fixture = [[8, 8, 1],
                        [2, 4, 8],
                        [7, 6, 4]]

    def test_sample_n(self):
        tf.random.set_seed(10402302)
        X = UniformInteger(0, 10)
        x = X.sample([3, 3])

        self.assertTrue(np.testing.assert_array_equal(x, self.fixture),
                        "UniformInteger sample inconsistent")

    def test_log_prob(self):
        X = UniformInteger(0, 10)
        lp = X.log_prob(self.fixture)
        self.assertSequenceEqual(lp.shape, [3, 3])
        self.assertAlmostEqual(np.sum(lp), -20.723267, places=6)

if __name__ == '__main__':
    unittest.main()
