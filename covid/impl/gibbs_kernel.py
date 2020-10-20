"""Gibbs sampling kernel"""
import inspect
import collections
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
from tensorflow_probability.python.experimental import unnest
from tensorflow_probability.python.internal import prefer_static

tfd = tfp.distributions  # pylint: disable=no-member
tfb = tfp.bijectors  # pylint: disable=no-member
mcmc = tfp.mcmc  # pylint: disable=no-member


class GibbsKernelResults(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple(
        "GibbsKernelResults", ["target_log_prob", "inner_results",],
    ),
):
    __slots__ = ()


def get_target_log_prob(results):
    """Fetches a target log prob from a results structure"""
    return unnest.get_innermost(results, "target_log_prob")


def update_target_log_prob(results, target_log_prob):
    """Puts a target log prob into a results structure"""
    if isinstance(results, GibbsKernelResults):
        replace_fn = unnest.replace_outermost
    else:
        replace_fn = unnest.replace_innermost
    return replace_fn(results, target_log_prob=target_log_prob)


def maybe_transform_value(tlp, state, kernel, direction):
    if not isinstance(kernel, tfp.mcmc.TransformedTransitionKernel):
        return tlp

    tlp_rank = prefer_static.rank(tlp)
    event_ndims = prefer_static.rank(state) - tlp_rank

    if direction == "forward":
        return tlp + kernel.bijector.inverse_log_det_jacobian(
            state, event_ndims=event_ndims
        )
    if direction == "inverse":
        return tlp - kernel.bijector.inverse_log_det_jacobian(
            state, event_ndims=event_ndims
        )
    raise AttributeError("`direction` must be `forward` or `inverse`")


class GibbsKernel(mcmc.TransitionKernel):
    def __init__(self, target_log_prob_fn, kernel_list, name=None):
        """Build a Gibbs sampling scheme from component kernels.

        :param target_log_prob_fn: a function that takes `state` arguments
                                   and returns the target log probability
                                   density.
        :param kernel_list: a list of tuples `(state_part_idx, kernel_make_fn)`.
                            `state_part_idx` denotes the index (relative to
                            positional args in `target_log_prob_fn`) of the
                            state the kernel updates.  `kernel_make_fn` takes
                            arguments `target_log_prob_fn` and `state`, returning
                            a `tfp.mcmc.TransitionKernel`.
        :returns: an instance of `GibbsKernel`
        """
        # Require to check if all kernel.is_calibrated is True
        self._parameters = dict(
            target_log_prob_fn=target_log_prob_fn, kernel_list=kernel_list, name=name,
        )

    @property
    def is_calibrated(self):
        return True

    @property
    def target_log_prob_fn(self):
        return self._parameters["target_log_prob_fn"]

    @property
    def kernel_list(self):
        return self._parameters["kernel_list"]

    @property
    def name(self):
        return self._parameters["name"]

    def one_step(self, current_state, previous_results, seed=None):
        """We iterate over the state elements, calling each kernel in turn.

        The `target_log_prob` is forwarded to the next `previous_results`
        such that each kernel has a current `target_log_prob` value.
        Transformations are automatically performed if the kernel is of
        type tfp.mcmc.TransformedTransitionKernel.

        In graph and XLA modes, the for loop should be unrolled.
        """
        if mcmc_util.is_list_like(current_state):
            next_state = list(current_state)
        else:
            next_state = [tf.convert_to_tensor(current_state)]

        current_state = [
            tf.convert_to_tensor(s, name="current_state") for s in current_state
        ]

        next_results = []
        untransformed_target_log_prob = previous_results.target_log_prob
        for i, (state_part_idx, kernel_fn) in enumerate(self.kernel_list):

            def target_log_prob_fn(state_part):
                next_state[
                    state_part_idx  # pylint: disable=cell-var-from-loop
                ] = state_part
                return self.target_log_prob_fn(*next_state)

            kernel = kernel_fn(target_log_prob_fn, next_state)

            previous_kernel_results = update_target_log_prob(
                previous_results.inner_results[i],
                maybe_transform_value(
                    tlp=untransformed_target_log_prob,
                    state=next_state[state_part_idx],
                    kernel=kernel,
                    direction="inverse",
                ),
            )

            next_state[state_part_idx], next_kernel_results = kernel.one_step(
                next_state[state_part_idx], previous_kernel_results, seed
            )

            next_results.append(next_kernel_results)
            untransformed_target_log_prob = maybe_transform_value(
                tlp=get_target_log_prob(next_kernel_results),
                state=next_state[state_part_idx],
                kernel=kernel,
                direction="forward",
            )

        return (
            next_state if mcmc_util.is_list_like(current_state) else next_state[0],
            GibbsKernelResults(
                target_log_prob=untransformed_target_log_prob,
                inner_results=next_results,
            ),
        )

    def bootstrap_results(self, current_state):

        if mcmc_util.is_list_like(current_state):
            current_state = list(current_state)
        else:
            current_state = [tf.convert_to_tensor(current_state)]
        current_state = [
            tf.convert_to_tensor(s, name="current_state") for s in current_state
        ]

        inner_results = []
        untransformed_target_log_prob = 0.0
        for state_part_idx, kernel_fn in self.kernel_list:

            def target_log_prob_fn(_):
                return self.target_log_prob_fn(*current_state)

            kernel = kernel_fn(target_log_prob_fn, current_state)
            kernel_results = kernel.bootstrap_results(current_state[state_part_idx])
            inner_results.append(kernel_results)
            untransformed_target_log_prob = maybe_transform_value(
                tlp=get_target_log_prob(kernel_results),
                state=current_state[state_part_idx],
                kernel=kernel,
                direction="forward",
            )

        return GibbsKernelResults(
            target_log_prob=untransformed_target_log_prob, inner_results=inner_results,
        )
