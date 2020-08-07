"""Gibbs sampler"""

import tensorflow_probability as tfp

__all__ = ["GibbsStep", "DeterministicScanKernel", "flatten_results"]

tfd = tfp.distributions  # pylint: disable=no-member
mcmc = tfp.mcmc  # pylint: disable=no-member


def flatten_results(results):
    """Results structures from nested Gibbs samplers sometimes
    need flattening for writing out purposes.
    """
    lst = []

    def recurse(r):
        for i in iter(r):
            if isinstance(i, list):
                for j in flatten_results(i):
                    yield j
            else:
                yield i

    return [r for r in recurse(results)]


def get_tlp(results):
    """Fetches a target log prob from a results structure"""
    if isinstance(results, list):
        return get_tlp(results[-1])
    else:
        return results.accepted_results.target_log_prob


def put_tlp(results, target_log_prob):
    """Puts a target log prob into a results structure"""
    if isinstance(results, list):
        results[0] = put_tlp(results[0], target_log_prob)
        return results
    else:
        accepted_results = results.accepted_results._replace(
            target_log_prob=target_log_prob
        )
        return results._replace(accepted_results=accepted_results)


def get_tlp_fn(kernel):
    if hasattr(kernel, "target_log_prob_fn"):
        return kernel.target_log_prob_fn
    else:
        return get_tlp_fn(kernel.inner_kernel)


def put_tlp_fn(kernel, target_log_prob_fn):
    if "target_log_prob_fn" in kernel.parameters:
        kernel.parameters["target_log_prob_fn"] = target_log_prob_fn
    else:
        put_tlp_fn(kernel.inner_kernel, target_log_prob_fn)


class GibbsStep(mcmc.TransitionKernel):
    def __init__(self, state_elem, inner_kernel, name=None):
        """Instantiates a Gibbs step

      :param state_elem: the index of the element in `state` to be sampled
      :param inner_kernel: the `tfp.mcmc.TransitionKernel` which operates on
                           the state element
      """
        self._parameters = dict(
            state_elem=state_elem,
            inner_kernel=inner_kernel,
            target_log_prob_fn=get_tlp_fn(inner_kernel),
            name=name,
        )

    @property
    def is_calibrated(self):
        return self._parameters["inner_kernel"].is_calibrated

    @property
    def state_elem(self):
        return self._parameters["state_elem"]

    @property
    def inner_kernel(self):
        return self._parameters["inner_kernel"]

    @property
    def target_log_prob_fn(self):
        return self._parameters["target_log_prob_fn"]

    def conditional_target_log_prob(self, state):
        """Closes over `state`, returning a function to
        calculate the conditional log prob for `state[state_elem]
        """

        def fn(state_part):
            state[self.state_elem] = state_part
            return self.target_log_prob_fn(*state)

        return fn

    def one_step(self, current_state, previous_results, seed=None):
        """Runs the Gibbs step.

        We close over the state, replacing the kernel target log
        probability function with the conditional log prob.
        """
        put_tlp_fn(self.inner_kernel, self.conditional_target_log_prob(current_state))
        next_state_part, next_results = self.inner_kernel.one_step(
            current_state[self.state_elem], previous_results, seed
        )
        current_state[self.state_elem] = next_state_part
        return current_state, next_results

    def bootstrap_results(self, current_state):
        put_tlp_fn(self.inner_kernel, self.conditional_target_log_prob(current_state))
        return self.inner_kernel.bootstrap_results(current_state[self.state_elem])


class DeterministicScanKernel(mcmc.TransitionKernel):
    def __init__(self, kernel_list, name=None):
        # Require to check if all kernel.is_calibrated is True
        self._parameters = dict(name=name, kernel_list=kernel_list)

    @property
    def is_calibrated(self):
        return True

    @property
    def kernel_list(self):
        return self._parameters["kernel_list"]

    def one_step(self, current_state, previous_results, seed=None):
        """We iterate over the state elements, calling each kernel in turn.

      The `target_log_prob` is forwarded to the next `previous_results`
      such that each kernel has a current `target_log_prob` value.  In graph 
      and XLA modes, the for loop should be unrolled.
      """
        next_results = previous_results  # Semantic sugar
        next_state = current_state
        for i, sampler in enumerate(self.kernel_list):
            next_state, next_results[i] = sampler.one_step(
                next_state, previous_results[i], seed
            )
            tlp = get_tlp(next_results[i])
            next_idx = (i + 1) % len(self.kernel_list)
            next_results[next_idx] = put_tlp(next_results[next_idx], tlp)
        return next_state, next_results

    def bootstrap_results(self, current_state):
        return [kernel.bootstrap_results(current_state) for kernel in self.kernel_list]
