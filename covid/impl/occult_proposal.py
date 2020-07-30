import tensorflow as tf
import tensorflow_probability as tfp

from covid.impl.UniformInteger import UniformInteger
from covid.impl.Categorical2 import Categorical2

tfd = tfp.distributions


def AddOccultProposal(events, n_max, t_range=None, dtype=tf.int32, name=None):
    if t_range is None:
        t_range = [0, events.shape[-2]]

    def m():
        """Select a metapopulation"""
        with tf.name_scope("m"):
            return UniformInteger(low=[0], high=[events.shape[0]], dtype=dtype)

    def t():
        """Select a timepoint"""
        with tf.name_scope("t"):
            return UniformInteger(low=[t_range[0]], high=[t_range[1]], dtype=dtype)

    def x_star():
        """Draw num to add"""
        return UniformInteger(low=[0], high=[n_max + 1], dtype=dtype)

    return tfd.JointDistributionNamed(dict(m=m, t=t, x_star=x_star), name=name)


def DelOccultProposal(events, topology, dtype=tf.int32, name=None):
    def m():
        """Select a metapopulation"""
        with tf.name_scope("m"):
            hot_meta = (
                tf.math.count_nonzero(
                    events[..., topology.target], axis=1, keepdims=True
                )
                > 0
            )
            hot_meta = tf.cast(tf.transpose(hot_meta), dtype=events.dtype)
            logits = tf.math.log(hot_meta)
            X = Categorical2(logits=logits, dtype=dtype, name="m")
            return X

    def t(m):
        """Draw timepoint"""
        with tf.name_scope("t"):
            metapops = tf.gather(events, m)
            hot_times = metapops[..., topology.target] > 0
            hot_times = tf.cast(hot_times, dtype=events.dtype)
            logits = tf.math.log(hot_times)
            return Categorical2(logits=logits, dtype=dtype, name="t")

    def x_star(m, t):
        """Draw num to delete"""
        with tf.name_scope("x_star"):
            indices = tf.stack([m, t, [topology.target]], axis=-1)
            max_occults = tf.gather_nd(events, indices)
            return UniformInteger(
                low=0, high=max_occults + 1, dtype=dtype, name="x_star"
            )

    return tfd.JointDistributionNamed(dict(m=m, t=t, x_star=x_star), name=name)
