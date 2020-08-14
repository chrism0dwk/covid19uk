"""Debugging tools"""

import tensorflow as tf
import tensorflow_probability as tfp


class DoNotUpdate(tfp.mcmc.TransitionKernel):
    def __init__(self, inner_kernel, name=None):
        """Prevents the update of a kernel for debug purposes"""
        self._parameters = dict(inner_kernel=inner_kernel, name=name)

    @property
    def inner_kernel(self):
        return self._parameters["inner_kernel"]

    @property
    def name(self):
        return self._parameters["name"]

    @property
    def is_calibrated(self):
        return True

    @property
    def parameters(self):
        return self._parameters

    def one_step(self, current_state, previous_results, seed=None):
        """Don't invoke inner_kernel.one_step, but return
        current state and results"""
        return current_state, previous_results

    def bootstrap_results(self, current_state):
        return self.inner_kernel.bootstrap_results(current_state)
