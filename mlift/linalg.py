import jax.numpy as np
import jax.lax as lax
import jax.api as api
from jax.lax.linalg import triangular_solve, cholesky
from mlift.math import log_diff_exp


def lu_triangular_solve(l, u, b):
    """Solve `l @ u @ x = b` where `l` and `u` are lower- and upper-triangular."""
    x = triangular_solve(l, b, left_side=True, lower=True)
    return triangular_solve(u, x, left_side=True, lower=False)


def jvp_cholesky_mtx_mult_by_vct(tangents, chol_mtx, vct):
    """Compute a Jacobian vector product for Cholesky of matrix multipled by a vector.

    For a function

        def f(mtx): return cholesky(mtx) @ vct

    this implements the forward-mode Jacobian vector product corresponding to

        einsum('ijk,jk->i', jacobian(f)(mtx), tangents)

    where `mtx` is a `(dim, dim)` shaped 2D array corresponding to a positive-definite
    matrix, `vct` is a `(dim,)` shaped 1D array and `tangents` is a `(dim, dim)` shaped
    symmetric 'tangent vector'.

    Uses forward-mode rule for Cholesky from https://arxiv.org/pdf/1602.07527.pdf
    """
    tmp = triangular_solve(
        chol_mtx,
        tangents,
        left_side=False,
        transpose_a=True,
        conjugate_a=True,
        lower=True,
    )
    tmp = triangular_solve(
        chol_mtx, tmp, left_side=True, transpose_a=False, conjugate_a=False, lower=True,
    )
    return chol_mtx @ (
        (np.tril(tmp) / (np.ones_like(chol_mtx) + np.identity(chol_mtx.shape[-1])))
        @ vct
    )


def tridiagonal_solve(a, b, c, d):
    """Solve a linear system `T @ x = d` for `x` where `T` is tridiagonal.

    Solves a linear system specified by a tridiagonal matrix with main diagonal
    `b`, lower-diagonal `a` and upper diagonal `c` for a vector right-hand side
    `d`. Equivalent to

        x = np.linalg.solve(np.diag(a, -1) + np.diag(b) + np.diag(c, 1), d)

    Args:
        a (array): lower diagonal of tridiagonal matrix, shape `(dim - 1,)`.
        b (array): main diagonal of tridiagonal matrix, shape `(dim,)`.
        c (array): upper diagonal of tridiagonal matrix, shape `(dim - 1,)`.
        d (array): vector right-hand-side, shape `(dim,)`.

    Returns:
        x (array): solution to system, shape `(dim,)`.
    """

    def forward(b_i_minus_1_d_i_minus_1, a_i_minus_1_b_i_c_i_minus_1_d_i):
        b_i_minus_1, d_i_minus_1 = b_i_minus_1_d_i_minus_1
        a_i_minus_1, b_i, c_i_minus_1, d_i = a_i_minus_1_b_i_c_i_minus_1_d_i
        w = a_i_minus_1 / b_i_minus_1
        b_i = b_i - w * c_i_minus_1
        d_i = d_i - w * d_i_minus_1
        return (b_i, d_i), (b_i, d_i)

    _, (b_, d_) = lax.scan(forward, (b[0], d[0]), (a, b[1:], c, d[1:]))

    b = np.concatenate((b[0:1], b_))
    d = np.concatenate((d[0:1], d_))

    def backward(x_i_plus_1, b_i_c_i_d_i):
        b_i, c_i, d_i = b_i_c_i_d_i
        x_i_plus_1 = (d_i - c_i * x_i_plus_1) / b_i
        return x_i_plus_1, x_i_plus_1

    _, x = lax.scan(backward, d[-1] / b[-1], (b[:-1], c, d[:-1]), reverse=True)

    return np.concatenate((x, d[-1:] / b[-1:]))


def tridiagonal_pos_def_log_det(a, b):
    """Compute the log-determinant of a tridiagonal positive-definite matrix.

    Computes the log-determinant for a tridiagonal matrix with main diagonal
    `b` (all positive) and lower- / upper- diagonal `a`. Equivalent to

        np.linalg.slogdet(diag(a, -1) + diag(b) + diag(a, 1))[1]

    Args:
        a (array): lower/upper diagonal of matrix, shape `(dim - 1,)`.
        b (array): main diagonal of matrix, shape `(dim,)`, all positive.

    Returns:
        Scalar corresponding to log-determinant.
    """

    def log_continuant_recursion(l_i_and_l_i_minus_1, a_i_and_b_i_plus_1):
        l_i, l_i_minus_1 = l_i_and_l_i_minus_1
        a_i, b_i_plus_1 = a_i_and_b_i_plus_1
        l_i_plus_1 = log_diff_exp(
            lax.log(b_i_plus_1) + l_i, 2 * lax.log(abs(a_i)) + l_i_minus_1
        )
        return (l_i_plus_1, l_i), None

    (l_n, _), _ = lax.scan(log_continuant_recursion, (lax.log(b[0]), 0), (a, b[1:]))

    return l_n

