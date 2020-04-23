import unittest
import numpy as np
import tensorflow as tf

from covid.impl.mcmc import make_event_time_move

class TestMakeEventTimeMove(unittest.TestCase):

    def setUp(self):
        self.events = np.random.randint(0, 15, [20, 100]).astype(np.float64)

    def test_matrix_where(self):
        from covid.impl.mcmc import matrix_where
        mw = matrix_where(self.events > 0.).numpy()
        tfw = tf.where(self.events > 0.).numpy()
        np.testing.assert_array_equal(mw, tfw)

    def test_sample(self):
        proposal = make_event_time_move(self.events, 0.005, 0.5, 5)
        @tf.function
        def sample():
            return proposal.sample()

        xstar = sample()
        print(xstar)

    def test_log_prob(self):
        proposal = make_event_time_move(self.events, 0.005, 0.5, 5)
        xstar = proposal.sample()

        lp = {}
        for i, k in zip(range(6), xstar.keys()):
            lp[k] = proposal.submodules[i].log_prob(xstar[k])
            print("Distribution:", proposal.submodules[i])
            print(lp[k])

        lp = tf.reduce_sum(proposal.log_prob(xstar))
        print(lp)

if __name__ == '__main__':
    unittest.main()
