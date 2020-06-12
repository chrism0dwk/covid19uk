import unittest
import pickle as pkl
import numpy as np

import tensorflow as tf

from covid.impl.event_time_proposal import _abscumdiff, EventTimeProposal, TransitionTopology


class TestAbsCumDiff(unittest.TestCase):

    def setUp(self):
        with open('../stochastic_sim_small.pkl','rb') as f:
            sim = pkl.load(f)
        self.events = np.stack([sim['events'][..., 0, 1],
                                sim['events'][..., 1, 2],
                                sim['events'][..., 2, 3]], axis=-1)
        self.events = np.transpose(self.events, axes=(1, 0, 2))
        self.initial_state = sim['state_init']

    def test_fwd_state(self):
        t = 19
        target_events = self.events[..., 1].copy()
        n_max = _abscumdiff(events=self.events, initial_state=self.initial_state, target_t=t,
                            target_id=1, bound_t=t, bound_id=2).numpy()
        # Numpy equivalent
        dNt_np = np.absolute(np.cumsum(self.events[..., 1] - self.events[..., 2], axis=-1))
        n_max_np = dNt_np[:, t] + self.initial_state[:, 2]
        np.testing.assert_array_equal(n_max.flatten(), n_max_np)

    def test_fwd_state_multi(self):
        t = [19, 20, 21]
        target_events = self.events[..., 1].copy()
        n_max = _abscumdiff(events=self.events, initial_state=self.initial_state,
                            target_t=t[0], target_id=1,
                            bound_t=t, bound_id=2).numpy()
        dNt_np = np.absolute(np.cumsum(self.events[..., 1] - self.events[..., 2], axis=-1))
        n_max_np = dNt_np[:, t] + self.initial_state[:, [2]]
        np.testing.assert_array_equal(n_max, n_max_np)

    def test_fwd_none(self):
        for t in range(self.events.shape[0]):
            n_max = _abscumdiff(events=self.events, initial_state=self.initial_state,
                                target_t=t, target_id=2,
                                bound_t=t, bound_id=-1).numpy()
            np.testing.assert_array_equal(n_max, np.zeros([self.events.shape[0], 1]))

    def test_bwd_state(self):
        t = 26
        n_max = _abscumdiff(events=self.events, initial_state=self.initial_state,
                            target_t=t, target_id=1,
                            bound_t=t-1, bound_id=0).numpy()
        dNt_np = np.absolute(np.cumsum(self.events[..., 1] - self.events[..., 0], axis=-1))
        n_max_np = dNt_np[:, t-1] + self.initial_state[:, 1]
        np.testing.assert_array_equal(n_max.flatten(), n_max_np)

    def test_bwd_none(self):
        for t in range(self.events.shape[0]):
            n_max = _abscumdiff(events=self.events, initial_state=self.initial_state,
                                target_t=t, target_id=0,
                                bound_t=t, bound_id=-1).numpy()
            np.testing.assert_array_equal(n_max, np.zeros([self.events.shape[0], 1]))


class TestEventTimeProposal(unittest.TestCase):
    def setUp(self):
        with open('../stochastic_sim_small.pkl', 'rb') as f:
            sim = pkl.load(f)
        self.events = np.stack([sim['events'][..., 0, 1],  # S->E
                                sim['events'][..., 1, 2],  # E->I
                                sim['events'][..., 2, 3]], axis=-1)  # I->R

        self.events = np.transpose(self.events, axes=(1, 0, 2)) # shape [M, T, K]
        hot_metapop = self.events[..., 1].sum(axis=-1) > 0
        self.events = self.events[hot_metapop]
        self.initial_state = sim['state_init'][hot_metapop]
        self.topology = TransitionTopology(prev=0, target=1, next=2)
        tf.random.set_seed(1)
        self.Q = EventTimeProposal(self.events, self.initial_state, self.topology,
                              3, 10)

    def test_event_time_proposal_sample(self):
        q = self.Q.sample()
        print(q)


if __name__ == '__main__':
    unittest.main()
