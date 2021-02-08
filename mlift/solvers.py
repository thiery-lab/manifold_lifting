import numpy as onp
from mici.states import _cache_key_func
from mici.errors import ConvergenceError


def euclidean_norm(vct):
    """Calculate the Euclidean (L-2) norm of a vector."""
    return (vct ** 2).sum() ** 0.5


def maximum_norm(vct):
    """Calculate the maximum (L-infinity) norm of a vector."""
    return abs(vct).max()


def jitted_solve_projection_onto_manifold_quasi_newton(
    state,
    state_prev,
    dt,
    system,
    constraint_tol=1e-9,
    position_tol=1e-8,
    divergence_tol=1e10,
    max_iters=50,
    norm=maximum_norm,
):
    """Symmetric quasi-Newton solver for projecting points onto manifold.

    Solves an equation of the form `r(λ) = c(q_ + c(q)ᵀλ) = 0` for the vector of
    Lagrange multipliers `λ` to project a point `q_` on to the manifold defined by the
    zero level set of `c`, with the projection performed with in the linear subspace
    defined by the rows of the Jacobian matrix `∂c(q)` evaluated at a previous point on
    the manifold `q`.

    The Jacobian of the residual function `r` is

        ∂r(λ) = ∂c(q_ + ∂c(q)ᵀλ) M⁻¹ ∂c(q)ᵀ

    such that the full Newton update

        λ = λ - ∂r(λ)⁻¹ r(λ)

    requires evaluating `∂c` on each iteration. The symmetric quasi-Newton iteration
    instead uses the approximation

        ∂c(q_ + ∂c(q)ᵀλ) ∂c(q)ᵀ ≈ ∂c(q) ∂c(q)ᵀ

    with the corresponding update

        λ = λ - (∂c(q) ∂c(q)ᵀ)⁻¹ r(λ)

    allowing a previously computed decomposition of the Gram matrix `∂c(q)∂c(q)ᵀ` to be
    used to solve the linear system in each iteration with no requirement to evaluate
    `∂c` on each iteration.

    Compared to the inbuilt solver in `mici.solvers` this version exploits the structure
    in the constraint Jacobian `∂c` and JIT compiles the iteration using JAX for better
    performance.
    """
    jacob_constr_blocks_prev = system.jacob_constr_blocks(state_prev)
    gram_components_prev = system.gram_components(state_prev)
    q_, mu, num_iters, norm_delta_q, error = system._quasi_newton_projection(
        state.pos,
        jacob_constr_blocks_prev,
        gram_components_prev,
        dt,
        constraint_tol,
        position_tol,
        divergence_tol,
        max_iters,
        norm,
    )
    num_iters = int(num_iters)
    if state._call_counts is not None:
        for method in [system.constr, system.lmult_by_pinv_jacob_constr]:
            key = _cache_key_func(system, method)
            if key in state._call_counts:
                state._call_counts[key] += num_iters
            else:
                state._call_counts[key] = num_iters
    if error < constraint_tol and norm_delta_q < position_tol:
        state.pos = onp.array(q_)
        if state.mom is not None:
            state.mom -= onp.asarray(mu)
        return state
    elif error > divergence_tol or onp.isnan(error):
        raise ConvergenceError(
            f"Quasi-Newton iteration diverged on iteration {num_iters}. "
            f"Last |c|={error:.1e}, |δq|={norm_delta_q}."
        )
    else:
        raise ConvergenceError(
            f"Quasi-Newton iteration did not converge in {num_iters} iterations. "
            f"Last |c|={error:.1e}, |δq|={norm_delta_q}."
        )


def jitted_solve_projection_onto_manifold_newton(
    state,
    state_prev,
    dt,
    system,
    constraint_tol=1e-9,
    position_tol=1e-8,
    divergence_tol=1e10,
    max_iters=50,
    norm=maximum_norm,
):
    """Newton solver for projecting points onto manifold.

    Solves an equation of the form `r(λ) = c(q_ + ∂c(q)ᵀλ) = 0` for the vector of
    Lagrange multipliers `λ` to project a point `q_` on to the manifold defined by the
    zero level set of `c`, with the projection performed with in the linear subspace
    defined by the rows of the Jacobian matrix `∂c(q)` evaluated at a previous point on
    the manifold `q`.

    The Jacobian of the residual function `r` is

        ∂r(λ) = ∂c(q_ + ∂c(q)ᵀλ) ∂c(q)ᵀ

    such that the Newton update is

        λ = λ - ∂r(λ)⁻¹ r(λ)

    which requires evaluating `∂c` on each iteration.

    Compared to the inbuilt solver in `mici.solvers` this version exploits the structure
    in the constraint Jacobian `∂c` and JIT compiles the iteration using JAX for better
    performance.
    """
    jacob_constr_blocks_prev = system.jacob_constr_blocks(state_prev)
    q_, mu, num_iters, norm_delta_q, error = system._newton_projection(
        state.pos,
        jacob_constr_blocks_prev,
        dt,
        constraint_tol,
        position_tol,
        divergence_tol,
        max_iters,
        norm,
    )
    num_iters = int(num_iters)
    if state._call_counts is not None:
        for method in [
            system.constr,
            system.jacob_constr_blocks,
            system.rmult_by_jacob_constr,
            system.lmult_by_inv_jacob_product,
        ]:
            key = _cache_key_func(system, method)
            if key in state._call_counts:
                state._call_counts[key] += num_iters
            else:
                state._call_counts[key] = num_iters
    if error < constraint_tol and norm_delta_q < position_tol:
        state.pos = onp.array(q_)
        if state.mom is not None:
            state.mom -= onp.asarray(mu)
        return state
    elif error > divergence_tol or onp.isnan(error):
        raise ConvergenceError(
            f"Newton iteration diverged on iteration {num_iters}. "
            f"Last |c|={error:.1e}, |δq|={norm_delta_q}."
        )
    else:
        raise ConvergenceError(
            f"Newton iteration did not converge in {num_iters} iterations. "
            f"Last |c|={error:.1e}, |δq|={norm_delta_q}."
        )
