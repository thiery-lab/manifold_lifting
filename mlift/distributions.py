from collections import namedtuple
import numpy as onp
import jax.numpy as np
from jax.scipy.special import gammaln, ndtr, ndtri, log_ndtr
from jax.core import ConcretizationTypeError
from mlift.math import log_diff_exp


RealInterval = namedtuple("RealInterval", ("lower", "upper"))
reals = RealInterval(-onp.inf, onp.inf)
positive_reals = RealInterval(0, onp.inf)
negative_reals = RealInterval(-onp.inf, 0)
# We ignore whether intervals bounds are open or closed as for distributions absolutely
# continuous wrt Lebesgue measure end-points of interval are zero-measure sets
# Therefore positive/non-negative and negative/non-positive equivalent here
nonnegative_reals = positive_reals
nonpositive_reals = negative_reals

import mlift.transforms as transforms


class Distribution:
    """Probability distribution with density with respect to Lebesgue measure."""

    def __init__(
        self,
        neg_log_dens,
        log_normalizing_constant,
        sample,
        support,
        from_standard_normal_transform=None,
    ):
        """
        Args:
            neg_log_dens (Callable[[ArrayLike], float]): Function returning the negative
                logarithm of a (potentially unnormalised) density function for the
                distribution with respect to the Lebesgue measure.
            log_normalizing_constant (ArrayLike): Logarithm of the normalising consant
                for density function defined by `neg_log_dens` such that
                    def dens(x): exp(-neg_log_dens(x) - log_normalizing_constant)
                is a normalized probability density function for the distribution.
            sample (Callable[[Generator, Tuple[int...]], ArrayLike]):
            support (object): Object defining support of distribution.
            from_standard_normal_transform (Callable[[ArrayLike], ArrayLike]): Function
                which given a random normal variate(s) outputs a variate(s) from the
                distribution represented by this object. Optional, may be `None`.
        """
        self._neg_log_dens = neg_log_dens
        self.log_normalizing_constant = log_normalizing_constant
        self.sample = sample
        self.support = support
        self.from_standard_normal_transform = from_standard_normal_transform

    def neg_log_dens(self, x, include_normalizing_constant=False):
        nld = self._neg_log_dens(x)
        if include_normalizing_constant:
            nld = nld + self.log_normalizing_constant
        if not (onp.isscalar(x) or x.shape == ()):
            nld = nld.sum()
        return nld


def pullback_distribution(distribution, transform):
    """Pullback a distribution through a differentiable transform.

    Given a distribution `μ` and (differentiable) transform `F` constructs a
    distribution `ν` such that `ν` is the pullback of `μ` under `F` or equivalently `μ`
    is the pushforward of `ν` under `F`, i.e. `F#ν = μ`.

    Args:
        distribution (Distribution): Distribution `μ` to pullback.
        transform (Transform): Transform `F` to pullback distribution through.

    Returns
        Distribution: Pullback distribution `ν`.
    """

    def transformed_neg_log_dens(u):
        x, det_dx_du = transform.forward_and_det_jacobian(u)
        return distribution.neg_log_dens(x) - np.log(det_dx_du)

    def transformed_sample(rng, shape=()):
        x_samples = distribution.sample(rng, shape)
        return onp.asarray(transform.backward(x_samples))

    assert (
        distribution.support == transform.image
    ), "Support of distribution does not match transform image"

    transformed_support = transform.domain

    if distribution.from_standard_normal_transform is not None:

        def transformed_from_standard_normal_transform(n):
            return transform.backward(distribution.from_standard_normal_transform(n))

    else:
        transformed_from_standard_normal_transform = None

    return Distribution(
        neg_log_dens=transformed_neg_log_dens,
        log_normalizing_constant=distribution.log_normalizing_constant,
        sample=transformed_sample,
        support=transformed_support,
        from_standard_normal_transform=transformed_from_standard_normal_transform,
    )


def uniform(lower, upper):
    """Construct uniform distribution with support on real-interval.

    Args:
        lower (float): Lower-bound of support.
        upper (float): Upper-bound of support.

    Returns:
        Distribution: Uniform distribution object.
    """

    def neg_log_dens(x):
        return 0

    log_normalizing_constant = np.log(upper - lower)

    def sample(rng, shape=()):
        return rng.uniform(low=lower, high=upper, size=shape)

    support = RealInterval(lower, upper)

    from_standard_normal_transform = transforms.standard_normal_to_uniform(lower, upper)

    return Distribution(
        neg_log_dens=neg_log_dens,
        log_normalizing_constant=log_normalizing_constant,
        sample=sample,
        support=support,
        from_standard_normal_transform=from_standard_normal_transform,
    )


def normal(location, scale):
    """Construct normal distribution with support on real-line.

    Args:
        location (float): Location parameter (mean of distribution).
        scale (float): Scale parameter (standard deviation of distribution).

    Returns:
        Distribution: Normal distribution object.
    """

    def neg_log_dens(x):
        return ((x - location) / scale) ** 2 / 2

    log_normalizing_constant = np.log(2 * np.pi) / 2 + np.log(scale)

    def sample(rng, shape=()):
        return rng.normal(loc=location, scale=scale, size=shape)

    from_standard_normal_transform = transforms.diagonal_affine_map(location, scale)

    return Distribution(
        neg_log_dens=neg_log_dens,
        log_normalizing_constant=log_normalizing_constant,
        sample=sample,
        support=reals,
        from_standard_normal_transform=from_standard_normal_transform,
    )


def log_normal(location, scale):
    """Construct log-normal distribution with support on positive reals.

    Args:
        location (float): Location parameter (mean of log of random variable).
        scale (float): Scale parameter (standard deviation of log of random variable).

    Returns:
        Distribution: Log-normal distribution object.
    """

    def neg_log_dens(x):
        return ((np.log(x) - location) / scale) ** 2 / 2 + np.log(x)

    log_normalizing_constant = np.log(2 * np.pi) / 2 + np.log(scale)

    def sample(rng, shape=()):
        return onp.exp(rng.normal(loc=location, scale=scale, size=shape))

    from_standard_normal_transform = transforms.ElementwiseMonotonicTransform(
        forward=lambda n: np.exp(location + scale * n),
        backward=lambda x: (np.log(x) - location) / scale,
        domain=reals,
        image=positive_reals
    )

    return Distribution(
        neg_log_dens=neg_log_dens,
        log_normalizing_constant=log_normalizing_constant,
        sample=sample,
        support=positive_reals,
        from_standard_normal_transform=from_standard_normal_transform,
    )


def half_normal(scale):
    """Construct half-normal distribution with support on non-negative reals.

    Args:
        scale (float): Scale parameter.

    Returns:
        Distribution: Half-normal distribution object.
    """

    def neg_log_dens(x):
        return (x / scale) ** 2 / 2

    log_normalizing_constant = np.log(np.pi / 2) / 2 + np.log(scale)

    def sample(rng, shape=()):
        return abs(rng.normal(loc=0, scale=scale, size=shape))

    from_standard_normal_transform = transforms.standard_normal_to_half_normal(scale)

    return Distribution(
        neg_log_dens=neg_log_dens,
        log_normalizing_constant=log_normalizing_constant,
        sample=sample,
        support=nonnegative_reals,
        from_standard_normal_transform=from_standard_normal_transform,
    )


def truncated_normal(location, scale, lower=-onp.inf, upper=onp.inf):
    """Construct truncated normal distribution with support on real interval.

    Args:
        location (float): Location parameter (mean of untruncated normal distribution).
        scale (float): Scale parameter (std. of untrunctated normal distribution).
        lower (float): Lower-bound of support.
        upper (float): Upper-bound of support.

    Returns:
        Distribution: Truncated normal distribution object.
    """

    def neg_log_dens(x):
        return ((x - location) / scale) ** 2 / 2

    log_normalizing_constant = (
        np.log(2 * np.pi) / 2
        + np.log(scale)
        + log_diff_exp(
            log_ndtr((upper - location) / scale), log_ndtr((lower - location) / scale)
        )
    )

    def sample(rng, shape=()):
        a = ndtr((lower - location) / scale)
        b = ndtr((upper - location) / scale)
        return ndtri(a + rng.uniform(size=shape) * (b - a)) * scale + location

    support = RealInterval(lower, upper)

    from_standard_normal_transform = transforms.standard_normal_to_truncated_normal(
        location, scale, lower, upper
    )

    return Distribution(
        neg_log_dens=neg_log_dens,
        log_normalizing_constant=log_normalizing_constant,
        sample=sample,
        support=support,
        from_standard_normal_transform=from_standard_normal_transform,
    )


def exponential(rate):
    """Construct exponential distribution with support on non-negative reals.

    Args:
        rate (float): Rate (inverse scale) parameter.

    Returns:
        Distribution: Exponential distribution object.
    """

    def neg_log_dens(x):
        return rate * x

    log_normalizing_constant = -np.log(rate)

    def sample(rng, shape=()):
        return rng.exponential(scale=1 / rate, size=shape)

    from_standard_normal_transform = transforms.standard_normal_to_exponential(rate)

    return Distribution(
        neg_log_dens=neg_log_dens,
        log_normalizing_constant=log_normalizing_constant,
        sample=sample,
        support=nonnegative_reals,
        from_standard_normal_transform=from_standard_normal_transform,
    )


def beta(shape_a, shape_b):
    """Beta distribution with support on unit interval.

    Args:
        shape_a: First shape parameter.
        shape_b: Second shape parameter.

    Returns:
        Distribution: Beta distribution object.
    """

    def neg_log_dens(x):
        return (1 - shape_a) * np.log(x) + (1 - shape_b) * np.log(1 - x)

    log_normalizing_constant = (
        gammaln(shape_a) + gammaln(shape_b) - gammaln(shape_a + shape_b)
    )

    def sample(rng, shape=()):
        return rng.beta(a=shape_a, b=shape_b, size=shape)

    support = RealInterval(0, 1)

    try:
        from_standard_normal_transform = transforms.standard_normal_to_beta(
            shape_a, shape_b
        )

    except (ValueError, ConcretizationTypeError):

        from_standard_normal_transform = None

    return Distribution(
        neg_log_dens=neg_log_dens,
        log_normalizing_constant=log_normalizing_constant,
        sample=sample,
        support=support,
        from_standard_normal_transform=from_standard_normal_transform,
    )


def gamma(shape, rate):
    """Gamma distribution with support on positive reals.

    Args:
        shape: Shape parameter.
        rate: Rate (inverse scale) parameter.

    Returns:
        Distribution: Gamma distribution object.
    """

    def neg_log_dens(x):
        return rate * x + (1 - shape) * np.log(x)

    log_normalizing_constant = gammaln(shape) - shape * np.log(rate)

    shape_param = shape

    def sample(rng, shape=()):
        return rng.gamma(shape=shape_param, scale=1 / rate, size=shape)

    return Distribution(
        neg_log_dens=neg_log_dens,
        log_normalizing_constant=log_normalizing_constant,
        sample=sample,
        support=positive_reals,
    )


def inverse_gamma(shape, scale):
    """Inverse gamma distribution with support on positive reals.

    Args:
        shape: Shape parameter.
        scale: Scale parameter.

    Returns:
        Distribution: Inverse gamma distribution object.
    """

    def neg_log_dens(x):
        return scale / x + (shape + 1) * np.log(x)

    log_normalizing_constant = gammaln(shape) - shape * np.log(scale)

    shape_param = shape

    def sample(rng, shape=()):
        return 1 / rng.gamma(shape=shape_param, scale=1 / scale, size=shape)

    return Distribution(
        neg_log_dens=neg_log_dens,
        log_normalizing_constant=log_normalizing_constant,
        sample=sample,
        support=positive_reals,
    )


def cauchy(location, scale):
    """Cauchy distribution with support on reals.

    Args:
        location (float): Location parameter.
        scale (float): Scale parameter.

    Returns:
        Distribution: Cauchy distribution object.
    """

    def neg_log_dens(x):
        return np.log1p(((x - location) / scale) ** 2)

    log_normalizing_constant = np.log(np.pi * scale)

    def sample(rng, shape=()):
        return location + rng.standard_cauchy(size=shape) * scale

    from_standard_normal_transform = transforms.standard_normal_to_cauchy(
        location, scale
    )

    return Distribution(
        neg_log_dens=neg_log_dens,
        log_normalizing_constant=log_normalizing_constant,
        sample=sample,
        support=reals,
        from_standard_normal_transform=from_standard_normal_transform,
    )


def half_cauchy(scale):
    """Half-Cauchy distribution with support on nonnegative reals.

    Args:
        scale (float): Scale parameter.

    Returns:
        Distribution: Half-Cauchy distribution object.
    """

    def neg_log_dens(x):
        return np.log1p((x / scale) ** 2)

    log_normalizing_constant = np.log(np.pi * scale / 2)

    def sample(rng, shape=()):
        return abs(rng.standard_cauchy(size=shape) * scale)

    from_standard_normal_transform = transforms.standard_normal_to_half_cauchy(scale)

    return Distribution(
        neg_log_dens=neg_log_dens,
        log_normalizing_constant=log_normalizing_constant,
        sample=sample,
        support=nonnegative_reals,
        from_standard_normal_transform=from_standard_normal_transform,
    )


def students_t(location, scale, dof):
    """Student's T distribution with support on reals.

    Args:
        location (float): Location parameter.
        scale (float): Scale parameter (positive).
        dof (float): Degrees of freedom parameter (positive).

    Returns:
        Distribution: Student's T distribution object.
    """

    def neg_log_dens(x):
        z = (x - location) / scale
        return (dof + 1) * np.log1p(z ** 2 / dof) / 2

    log_normalizing_constant = (
        gammaln(dof / 2)
        - gammaln((dof + 1) / 2)
        + np.log(np.pi * dof) / 2
        + np.log(scale)
    )

    def sample(rng, shape=()):
        return location + rng.standard_t(df=dof, size=shape) * scale

    try:
        from_standard_normal_transform = transforms.standard_normal_to_students_t(
            location, scale, dof
        )

    except (ValueError, ConcretizationTypeError):

        from_standard_normal_transform = None

    return Distribution(
        neg_log_dens=neg_log_dens,
        log_normalizing_constant=log_normalizing_constant,
        sample=sample,
        support=reals,
        from_standard_normal_transform=from_standard_normal_transform,
    )


def half_students_t(scale, dof):
    """Half-Student's T distribution with support on nogative reals.

    Args:
        scale (float): Scale parameter (positive).
        dof (float): Degrees of freedom parameter (positive).

    Returns:
        Distribution: Half-Student's T distribution object.
    """

    def neg_log_dens(x):
        z = x / scale
        return (dof + 1) * np.log1p(z ** 2 / dof) / 2

    log_normalizing_constant = (
        gammaln(dof / 2)
        - gammaln((dof + 1) / 2)
        + np.log(np.pi * dof) / 2
        + np.log(scale / 2)
    )

    def sample(rng, shape=()):
        return abs(rng.standard_t(df=dof, size=shape) * scale)

    return Distribution(
        neg_log_dens=neg_log_dens,
        log_normalizing_constant=log_normalizing_constant,
        sample=sample,
        support=nonnegative_reals,
    )

