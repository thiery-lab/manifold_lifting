import numpy as onp
import jax.numpy as np
import jax.api as api
from jax.scipy.special import ndtr, ndtri, logit, expit
from mlift.math import (
    standard_cauchy_cdf,
    standard_cauchy_icdf,
    standard_students_t_2_cdf,
    standard_students_t_2_icdf,
    standard_students_t_4_cdf,
    standard_students_t_4_icdf,
)
from mlift.distributions import RealInterval, reals, nonnegative_reals


class ElementwiseMonotonicTransform:
    def __init__(self, forward, backward, domain, image, val_and_grad_forward=None):
        self._forward = forward
        self._backward = backward
        self.domain = domain
        self.image = image
        if val_and_grad_forward is None:
            val_and_grad_forward = api.value_and_grad(forward)
        self._val_and_grad_forward = val_and_grad_forward

    def forward(self, u):
        return self._forward(u)

    def backward(self, x):
        return self._backward(x)

    def forward_and_det_jacobian(self, u):
        if onp.isscalar(u) or u.shape == ():
            return self._val_and_grad_forward(u)
        else:
            x, dx_du = api.vmap(self._val_and_grad_forward)(u)
            return x, dx_du.sum()

    def __call__(self, u):
        return self._forward(u)


def unbounded_to_lower_bounded(lower):
    """Construct transform from reals to lower-bounded interval.

    Args:
        lower (float): Lower-bound of image of transform.
    """

    return ElementwiseMonotonicTransform(
        forward=lambda u: np.exp(u) + lower,
        backward=lambda x: np.log(x - lower),
        domain=reals,
        image=RealInterval(lower, onp.inf),
    )


def unbounded_to_upper_bounded(upper):
    """Construct transform from reals to upper-bounded interval.

    Args:
        upper (float): Upper-bound of image of transform.
    """
    return ElementwiseMonotonicTransform(
        forward=lambda u: upper - np.exp(u),
        backward=lambda x: np.log(upper - x),
        domain=reals,
        image=RealInterval(-onp.inf, upper),
    )


def unbounded_to_lower_and_upper_bounded(lower, upper):
    """Construct transform from reals to bounded interval.

    Args:
        lower (float): Lower-bound of image of transform.
        upper (float): Upper-bound of image of transform.
    """
    return ElementwiseMonotonicTransform(
        forward=lambda u: lower + (upper - lower) * expit(np.asarray(u, np.float64)),
        backward=lambda x: logit((np.asarray(x, np.float64) - lower) / (upper - lower)),
        domain=reals,
        image=RealInterval(lower, upper),
    )


def diagonal_affine_map(location, scale):

    return ElementwiseMonotonicTransform(
        forward=lambda x: location + scale * x,
        backward=lambda y: (y - location) / scale,
        domain=reals,
        image=reals,
        val_and_grad_forward=lambda x: (location + scale * x, scale),
    )


def standard_normal_to_uniform(lower, upper):

    return ElementwiseMonotonicTransform(
        forward=lambda n: lower + ndtr(n) * (upper - lower),
        backward=lambda u: ndtri(u - lower / (upper - lower)),
        domain=reals,
        image=RealInterval(lower, upper),
    )


def standard_normal_to_exponential(rate):

    return ElementwiseMonotonicTransform(
        forward=lambda n: -np.log(ndtr(n)) / rate,
        backward=lambda e: ndtri(np.exp(-e * rate)),
        domain=reals,
        image=nonnegative_reals,
    )


def standard_normal_to_cauchy(location, scale):
    return ElementwiseMonotonicTransform(
        forward=lambda n: location + standard_cauchy_icdf(ndtr(n)) * scale,
        backward=lambda c: ndtri(standard_cauchy_cdf((c - location) / scale)),
        domain=reals,
        image=reals,
    )


def standard_normal_to_half_normal(scale):

    return ElementwiseMonotonicTransform(
        forward=lambda n: ndtri((ndtr(n) + 1) / 2) * scale,
        backward=lambda h: ndtri(2 * ndtr(h / scale) - 1),
        domain=reals,
        image=nonnegative_reals,
    )


def standard_normal_to_half_cauchy(scale):

    return ElementwiseMonotonicTransform(
        forward=lambda n: standard_cauchy_icdf((ndtr(n) + 1) / 2) * scale,
        backward=lambda h: ndtri(2 * standard_cauchy_cdf(h / scale) - 1),
        domain=reals,
        image=nonnegative_reals,
    )


def standard_normal_to_truncated_normal(location, scale, lower, upper):

    a = ndtr((lower - location) / scale)
    b = ndtr((upper - location) / scale)
    return ElementwiseMonotonicTransform(
        forward=lambda n: ndtri(a + ndtr(n) * (b - a)) * scale + location,
        backward=lambda t: ndtri((ndtr((t - location) / scale) - a) / (b - a)),
        domain=reals,
        image=RealInterval(lower, upper),
    )


def standard_normal_to_beta(shape_a, shape_b):

    if shape_b == 1:

        def icdf(u):
            return u ** (1 / shape_a)

        def cdf(x):
            return x ** shape_a

    elif shape_a == 1:

        def icdf(u):
            return 1 - (1 - u) ** (1 / shape_b)

        def cdf(x):
            return 1 - (1 - x) ** shape_b

    else:

        raise ValueError("Transform only defined for shape_a == 1 or shape_b == 1")

    return ElementwiseMonotonicTransform(
        forward=lambda n: icdf(ndtr(n)),
        backward=lambda x: ndtri(cdf(x)),
        domain=reals,
        image=RealInterval(0, 1),
    )


def standard_normal_to_students_t(location, scale, dof):

    if dof == 1:

        return standard_normal_to_cauchy(location, scale)

    elif dof == 2:

        return ElementwiseMonotonicTransform(
            forward=lambda n: location + standard_students_t_2_icdf(ndtr(n)) * scale,
            backward=lambda t: ndtri((standard_students_t_2_cdf(t - location) / scale)),
            domain=reals,
            image=reals,
        )

    elif dof == 4:

        return ElementwiseMonotonicTransform(
            forward=lambda n: location + standard_students_t_4_icdf(ndtr(n)) * scale,
            backward=lambda t: ndtri((standard_students_t_4_cdf(t - location) / scale)),
            domain=reals,
            image=reals,
        )

    else:

        raise ValueError("Transform only defined for dof in {1, 2, 4}")

