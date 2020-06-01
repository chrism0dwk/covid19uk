import unittest
import pickle as pkl
import numpy as np

from covid.impl.event_time import _max_free_events


class TestMaxFreeEvents(unittest.TestCase):

    def setUp(self):
        with open('../stochastic_sim_small.pkl','rb') as f:
            sim = pkl.load(f)
        self.events = np.stack([sim['events'][..., 0, 1],
                                sim['events'][..., 1, 2],
                                sim['events'][..., 2, 3]], axis=-1)
        self.initial_state = sim['state_init']

    def test_fwd_state(self):
        t = 19
        target_events = self.events[..., 1].copy()
        n_max = _max_free_events(events=self.events, target_t=t, target_id=1,
                                 constraint_t=t, constraint_id=2).numpy()
        np.testing.assert_array_equal(n_max, [1., 2., 4., 1., 0., 0., 0., 0., 0., 0.])

        target_events[t, :] -= n_max
        target_events[t+1, :] += n_max
        state = np.cumsum(target_events, axis=0) + self.initial_state[:, 1]
        self.assertTrue(np.all(target_events >= 0))
        self.assertTrue(np.all(state >= 0))

    def test_fwd_state_multi(self):
        t = [19, 20, 21]
        target_events = self.events[..., 1].copy()
        n_max = _max_free_events(events=self.events, initial_state=self.initial_state,
                                 target_t=t[0], target_id=1,
                                 constraint_t=t, constraint_id=2).numpy()
        np.testing.assert_array_equal(n_max, [1., 2., 4., 1., 0., 0., 0., 0., 0., 0.])

        target_events[np.min(t), :] -= n_max
        target_events[np.max(t), :] += n_max
        state = np.cumsum(target_events, axis=0) + self.initial_state[:, 1]
        self.assertTrue(np.all(target_events >= 0))
        self.assertTrue(np.all(state >= 0))

    def test_fwd_none(self):
        for t in range(self.events.shape[0]):
            n_max = _max_free_events(events=self.events, target_t=t, target_id=2,
                                     constraint_t=t, constraint_id=-1).numpy()
            np.testing.assert_array_equal(n_max, self.events[t, :, 2])

    def test_bwd_state(self):
        t = 26
        target_events = self.events[..., 1].copy()
        constraining_events = self.events[..., 0].copy()
        n_max = _max_free_events(events=self.events, target_t=t, target_id=1,
                                 constraint_t=t-1, constraint_id=0).numpy()
        np.testing.assert_array_equal(n_max, [0., 5., 3., 4., 0., 0., 0., 0., 0., 0.])

        # Assert moving events doesn't invalidate the state
        print('Before move', target_events[t])
        target_events[t, :] -= n_max
        target_events[t-1, :] += n_max
        print('After move', target_events[t])
        state = np.cumsum(target_events, axis=0)
        self.assertTrue(np.all(target_events >= 0))
        self.assertTrue(np.all(state >= 0))

    def test_bwd_none(self):
        for t in range(self.events.shape[0]):
            n_max = _max_free_events(events=self.events, target_t=t, target_id=0,
                                     constraint_t=t, constraint_id=-1).numpy()
            np.testing.assert_array_equal(n_max, self.events[t, :, 0])


if __name__ == '__main__':
    unittest.main()
