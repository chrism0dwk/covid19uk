import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import reparameterization
tfd = tfp.distributions


class KCategorical(tfd.Distribution):
    def __init__(self,
                 k,
                 probs,
                 validate_args=False,
                 allow_nan_stats=True,
                 name="Categorical"):
        """K-Categorical distribution

        Given a set of items indexed $1,...,n$ with weights $w_1,\dots,w_n$,
        sample $k$ indices without replacement.
        :param k: the number of indices to sample
        :param probs: the (normalized) probability vector
        :param validate_args: Whether to validate args
        :param allow_nan_stats: allow nan stats
        :param name: name of the distribution
        """
        parameters = dict(locals())
        self.probs = probs
        self.logits = tf.math.log(probs)
        dtype = tf.int32

        with tf.name_scope(name) as name:
            super(KCategorical, self).__init__(
                dtype=dtype,
                reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
                parameters=parameters,
                name=name)

    def _sample_n(self, n, seed=None):
        g = tfd.Gumbel(0., 1.).sample(self.logits.shape, seed=seed)
        z = g + self.logits
        x = tf.argsort(z)
        return x[:self.parameters['k']]

    def _log_prob(self, x):
        n = self.logits.shape
        k = x.shape
        wz = tf.gather(self.probs, x)
        W = tf.cumsum(wz, reverse=True)
        return tf.reduce_sum(wz - tf.math.log(W))


if __name__=='__main__':
    probs = tf.constant([1, 0, 0, 1, 1, 0, 1], dtype=tf.float32)
    probs = probs/tf.reduce_sum(probs)
    X = KCategorical(3, probs)
    x = X.sample()
    print(x)
    lp = X.log_prob(x)
    print(lp)
