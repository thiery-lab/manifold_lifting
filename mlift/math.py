import numpy as onp
import jax.lax as lax
import jax.numpy as np
from jax.scipy.linalg import expm as _expm


def log1m_exp(val):
    """Numerically stable implementation of `log(1 - exp(val))`."""
    return lax.cond(
        lax.gt(val, lax.log(2.0)),
        lambda _: lax.log(-lax.expm1(val)),
        lambda _: lax.log1p(-lax.exp(val)),
        operand=None,
    )


def log_diff_exp(val1, val2):
    """Numerically stable implementation of `log(exp(val1) - exp(val2))`."""
    return val1 + log1m_exp(val2 - val1)


def standard_cauchy_cdf(c):
    """Cumulative distribution function of standard Cauchy."""
    return np.arctan(c) / np.pi + 0.5


def standard_cauchy_icdf(u):
    """Inverse cumulative distribution function of standard Cauchy."""
    return np.tan(np.pi * (u - 0.5))


def standard_students_t_2_cdf(t):
    """Cumulative distribution function of standard Student's T with dof = 2."""
    return 0.5 + t / (2 * np.sqrt(2 + t ** 2))


def standard_students_t_2_icdf(u):
    """Inverse cumulative distribution function of standard Student's T with dof = 4."""
    return 2 * (u - 0.5) * np.sqrt(2 / (4 * u * (1 - u)))


def standard_students_t_4_cdf(t):
    """Cumulative distribution function of standard Student's t with dof = 4."""
    t_sq = t ** 2
    return 0.5 + (3 / 8) * (t / np.sqrt(1 + t_sq / 4)) * (
        1 - (t_sq / (1 + t_sq / 4)) / 12
    )


def standard_students_t_4_icdf(u):
    """Inverse cumulative distribution function of standard Student's t with dof = 4."""
    sqrt_α = np.sqrt(4 * u * (1 - u))
    return 2 * np.sign(u - 0.5) * np.sqrt(np.cos(np.arccos(sqrt_α) / 3) / sqrt_α - 1)


def expm(a):
    """Matrix exponential function with special handling of 1x1, diagonal and 2x2 cases.

    Args:
        a (ArrayLike): Array specifying matrix to be exponentiated. Either a square 2D
            array directly representing matrix or a scalar, assumed to represent a 1x1
            matrix or a 1D array, assumed to represent the diagonal of a square diagonal
            matrix.

    Returns:
        ArrayLike: Matrix exponential of matrix represented by `a`.
    """
    if np.isscalar(a) or a.ndim <= 1 or a.shape == (1, 1):
        return np.exp(a)
    elif a.shape == (2, 2):
        a_00, a_01, a_10, a_11 = a[0, 0], a[0, 1], a[1, 0], a[1, 1]
        b = np.exp((a_00 + a_11) / 2)
        c = a_00 - a_11
        d = (c ** 2 + 4 * a_10 * a_01) ** 0.5
        return lax.cond(
            np.less_equal(abs(d), onp.finfo(a.dtype).eps),
            lambda _: b * np.array([[1 + c / 2, a_01], [a_10, 1 - c / 2]]),
            lambda _: b
            * np.array(
                [
                    [
                        np.cosh(d / 2) + c * np.sinh(d / 2) / d,
                        2 * a_01 * np.sinh(d / 2) / d,
                    ],
                    [
                        2 * a_10 * np.sinh(d / 2) / d,
                        np.cosh(d / 2) - c * np.sinh(d / 2) / d,
                    ],
                ]
            ),
            operand=None,
        )
    else:
        return _expm(a)
