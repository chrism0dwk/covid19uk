"""The UniformInteger distribution class"""

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util

tfd = tfp.distributions


class UniformInteger(tfd.Distribution):
    def __init__(self, low=0, high=1, validate_args=False,
                 allow_nan_stats=True, dtype=tf.int32, name='UniformInteger'):
        """Initialise a UniformInteger random variable on `[low, high)`.

        Args:
          low: Integer tensor, lower boundary of the output interval. Must have
           `low <= high`.
          high: Integer tensor, _inclusive_ upper boundary of the output
           interval.  Must have `low <= high`.
          validate_args: Python `bool`, default `False`. When `True` distribution
           parameters are checked for validity despite possibly degrading runtime
           performance. When `False` invalid inputs may silently render incorrect
           outputs.
           allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
           (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
           result is undefined. When `False`, an exception is raised if one or
           more of the statistic's batch members are undefined.
           dtype: the dtype of the output variates
          name: Python `str` name prefixed to Ops created by this class.

        Raises:
          InvalidArgument if `low > high` and `validate_args=False`.
        """
        parameters = dict(locals())
        with tf.name_scope(name) as name:
            self._low = tf.cast(low, name='low', dtype=dtype)
            self._high = tf.cast(high, name='high', dtype=dtype)
            super(UniformInteger, self).__init__(
                dtype=dtype,
                reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
                parameters=parameters,
                name=name)

    @staticmethod
    def _param_shapes(sample_shape):
        return dict(
            zip(('low', 'high'),
                ([tf.convert_to_tensor(sample_shape, dtype=tf.int32)] * 2)))

    @classmethod
    def _params_event_ndims(cls):
        return dict(low=0, high=0)

    @property
    def low(self):
        """Lower boundary of the output interval."""
        return self._low

    @property
    def high(self):
        """Upper boundary of the output interval."""
        return self._high

    def range(self, name='range'):
        """`high - low`."""
        with self._name_and_control_scope(name):
            return self._range()

    def _range(self, low=None, high=None):
        low = self.low if low is None else low
        high = self.high if high is None else high
        return high - low

    def _batch_shape_tensor(self, low=None, high=None):
        return tf.broadcast_dynamic_shape(
            tf.shape(self.low if low is None else low),
            tf.shape(self.high if high is None else high))

    def _batch_shape(self):
        return tf.broadcast_static_shape(
            self.low.shape,
            self.high.shape)

    def _event_shape_tensor(self):
        return tf.constant([], dtype=tf.int32)

    def _event_shape(self):
        return tf.TensorShape([])

    def _sample_n(self, n, seed=None):
        low = tf.convert_to_tensor(self.low)
        high = tf.convert_to_tensor(self.high)
        shape = tf.concat([[n], self._batch_shape_tensor(
            low=low, high=high)], 0)
        samples = samplers.uniform(shape=shape, dtype=tf.float32, seed=seed)
        return low + tf.cast(
            tf.cast(self._range(low=low, high=high), tf.float32) * samples,
            self.dtype)

    def _prob(self, x):
        low = tf.cast(self.low, tf.float32)
        high = tf.cast(self.high, tf.float32)
        x = tf.cast(x, dtype=tf.float32)

        return tf.where(
            tf.math.is_nan(x),
            x,
            tf.where(
                (x < low) | (x >= high),
                tf.zeros_like(x),
                tf.ones_like(x) / self._range(low=low, high=high)))

    def _log_prob(self, x):
        return tf.math.log(self._prob(x))
