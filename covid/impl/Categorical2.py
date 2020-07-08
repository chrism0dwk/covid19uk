"""Categorical2 corrects a bug in the tfd.Categorical.log_prob"""

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.distributions.categorical import (
    _broadcast_cat_event_and_params,
)

tfd = tfp.distributions

# Todo remove this class when https://github.com/tensorflow/tensorflow/issues/40606
#   is fixed
class Categorical2(tfd.Categorical):
    """Done to override the faulty log_prob in tfd.Categorical due to
       https://github.com/tensorflow/tensorflow/issues/40606"""

    def _log_prob(self, k):
        logits = self.logits_parameter()
        if self.validate_args:
            k = distribution_util.embed_check_integer_casting_closed(
                k, target_dtype=self.dtype
            )
        k, logits = _broadcast_cat_event_and_params(
            k, logits, base_dtype=dtype_util.base_dtype(self.dtype)
        )
        logits_normalised = tf.math.log(tf.math.softmax(logits))
        return tf.gather(logits_normalised, k, batch_dims=1)
