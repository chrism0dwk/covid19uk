import unittest
import numpy as np
import tensorflow as tf

from covid.impl.util import make_transition_rate_matrix

state2b3c3s = [[[[0., 0.3996, 0.  ],
                 [0., 0.,     0.14],
                 [0., 0.,     0.  ]],
                [[0., 0.398,  0.  ],
                 [0., 0.,     0.14],
                 [0., 0.,     0.  ]]],
               [[[0., 0.2772, 0.  ],
                 [0., 0.,     0.98],
                 [0., 0.,     0.  ]],
                [[0., 2.25,   0.  ],
                 [0., 0.,     2.1 ],
                 [0., 0.,     0.  ]]]]

class TestRateUtils(unittest.TestCase):

    def test_transition_rate_matrix2b3c3s(self):
        state = tf.constant([[[999, 1, 0],
                              [199, 1, 0]],
                             [[99, 7, 894],
                              [75, 15, 110]]], dtype=tf.float32)

        si = 0.4 * state[..., 0] * state[..., 1] / \
             tf.reduce_sum(state, axis=-1)  # S->I rate
        ir = 0.14 * state[..., 1]  # I->R rate

        trm = make_transition_rate_matrix([si, ir], [[0, 1], [1, 2]], state)
        np.testing.assert_array_almost_equal(trm.numpy(), state2b3c3s, decimal=5)


if __name__ == '__main__':
    unittest.main()
