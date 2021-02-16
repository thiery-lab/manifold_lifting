import jax.lax as lax
import jax.numpy as np


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
