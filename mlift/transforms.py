import jax.numpy as np
from jax.scipy.special import ndtr, ndtri


def normal_to_uniform(n):
    """Transform standard normal variate to standard uniform variate."""
    return ndtr(n)


def uniform_to_normal(u):
    """Transform standard uniform variate to standard normal variate."""
    return ndtri(u)


def students_t_4_to_uniform(t):
    """Transform standard Student's t variate with ν = 4 to standard uniform variate."""
    t_sq = t ** 2
    return 0.5 + (3 / 8) * (t / np.sqrt(1 + t_sq / 4)) * (
        1 - (t_sq / (1 + t_sq / 4)) / 12
    )


def uniform_to_students_t_4(u):
    """Transform standard uniform variate to standard Student's t variate with ν = 4."""
    sqrt_α = np.sqrt(4 * u * (1 - u))
    return 2 * np.sign(u - 0.5) * np.sqrt(np.cos(np.arccos(sqrt_α) / 3) / sqrt_α - 1)


def students_t_2_to_uniform(t):
    """Transform standard Student's t variate with ν = 2 to standard uniform variate."""
    return 0.5 + t / (2 * np.sqrt(2 + t**2))


def uniform_to_students_t_2(u):
    """Transform standard uniform variate to standard Student's t variate with ν = 2."""
    sqrt_α = np.sqrt(4 * u * (1 - u))
    return 2 * (u - 0.5) * np.sqrt(2 / sqrt_α)


def cauchy_to_uniform(c):
    """Transform standard Cauchy variate to standard uniform variate."""
    return np.arctan(c) / np.pi + 0.5


def uniform_to_cauchy(u):
    """Transform standard uniform variate to standard normal variate."""
    return np.tan(np.pi * (u - 0.5))


def normal_to_students_t_4(n):
    """Transform standard normal variate to standard Student's t variate with ν = 4."""
    return uniform_to_students_t_4(normal_to_uniform(n))


def normal_to_students_t_2(n):
    """Transform standard normal variate to standard Student's t variate with ν = 2."""
    return uniform_to_students_t_2(normal_to_uniform(n))


def normal_to_cauchy(n):
    """Transform standard normal variate to standard Cauchy variate."""
    return uniform_to_cauchy(normal_to_uniform(n))


def normal_to_half_cauchy(n):
    """Transform standard normal variate to half-Cauchy variate."""
    return uniform_to_cauchy((normal_to_uniform(n) + 1) / 2)


def normal_to_half_normal(n):
    """Transform standard normal variate to half-normal variate."""
    return uniform_to_normal((normal_to_uniform(n) + 1) / 2)


def students_t_4_to_normal(t):
    """Transform standard normal variate to standard Student's t variate with ν = 4."""
    return uniform_to_normal(students_t_4_to_uniform(t))


def students_t_2_to_normal(t):
    """Transform standard normal variate to standard Student's t variate with ν = 2."""
    return uniform_to_normal(students_t_2_to_uniform(t))


def cauchy_to_normal(c):
    """Transform standard Cauchy variate to standard normal variate."""
    return uniform_to_normal(cauchy_to_uniform(c))


def half_cauchy_to_normal(h):
    """Transform half-Cauchy variate to standard normal variate."""
    return uniform_to_normal(2 * cauchy_to_uniform(h) - 1)


def half_normal_to_normal(h):
    """Transform half-normal variate to standard normal variate."""
    return uniform_to_normal(2 * normal_to_uniform(h) - 1)

