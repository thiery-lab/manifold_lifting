from mici.systems import System
from mici.matrices import IdentityMatrix
from mici.states import cache_in_state, cache_in_state_with_aux

from jax.config import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

import jax.api as api
import jax.lax as lax
import jax.numpy as np
import jax.scipy.linalg as sla
from jax.lax.linalg import triangular_solve, cholesky
import numpy as onp

from mlift.linalg import (
    lu_triangular_solve,
    jvp_cholesky_mtx_mult_by_vct,
    tridiagonal_solve,
    tridiagonal_pos_def_log_det,
)
from mlift.solvers import maximum_norm


def standard_normal_neg_log_dens(q):
    """Unnormalised negative log density of standard normal vector."""
    return 0.5 * onp.sum(q ** 2)


def standard_normal_grad_neg_log_dens(q):
    """Gradient and value of negative log density of standard normal vector."""
    return q, 0.5 * onp.sum(q ** 2)


def convert_to_numpy_pytree(jax_pytree):
    """Recursively convert 'pytree' of JAX arrays to NumPy arrays."""
    if isinstance(jax_pytree, np.DeviceArray):
        return onp.asarray(jax_pytree)
    elif isinstance(jax_pytree, (float, int, complex, bool, type(None))):
        return jax_pytree
    elif isinstance(jax_pytree, tuple):
        return tuple(convert_to_numpy_pytree(subtree) for subtree in jax_pytree)
    elif isinstance(jax_pytree, list):
        return [convert_to_numpy_pytree(subtree) for subtree in jax_pytree]
    elif isinstance(jax_pytree, dict):
        return {k: convert_to_numpy_pytree(v) for k, v in jax_pytree.items()}
    else:
        raise ValueError(f"Unknown jax_pytree node type {type(jax_pytree)}")


def construct_mici_system_neg_log_dens_functions(jax_neg_log_dens):
    """Construct functions to initialise system from JAX negative log density function.

    Given a negative log density function defined using JAX primitives, constructs `jit`
    transformed versions of the function and its gradient and then wraps these into
    functions with the required interface for passing to the initialiser of a Mici
    system class, specifically ensuring the values outputted by the functions are NumPy
    `ndarray` objects rather than JAX `DeviceArray` or `Buffer` instances and returning
    both the gradient and value from the gradient function (in that order).

    Args:
        jax_neg_log_dens (Callable): Function accepting a single array argument and
            returning a scalar corresponding to the negative logarithm of the target
            density. Should be constructed using only JAX primitives (e.g. from the
            `jax.numpy` API) so that JAX transforms can be applied.

    Returns:
        A tuple `(neg_log_dens, grad_neg_log_dens)`, with `neg_log_dens` a function
        accepting a single array argument and returning a NumPy scalar corresponding to
        the negative logarithm of the target density, and `grad_neg_log_dens` a function
        accepting a single array argument and returning a tuple `(grad, val)` with
        `grad` a NumPy array corresponding to the gradient of the negative logarithm of
        the target density with respect o the input latent vector, and `val` its value.
    """

    jitted_neg_log_dens = api.jit(jax_neg_log_dens)
    jitted_val_and_grad_neg_log_dens = api.jit(api.value_and_grad(jax_neg_log_dens))

    def neg_log_dens(q):
        return onp.asarray(jitted_neg_log_dens(q))

    def grad_neg_log_dens(q):
        val, grad = jitted_val_and_grad_neg_log_dens(q)
        return onp.asarray(grad), onp.asarray(val)

    return neg_log_dens, grad_neg_log_dens


def construct_state_space_model_generators(
    generate_params, generate_x_0, forward_func, observation_func
):
    """Construct functions to generate obs. and state sequences for state space models.

    Args:
        generate_params (Callable[[ArrayLike, Dict], Dict]): Function which generates a
            dictionary of model parameters given a 1D array of unbounded global latent
            variables and data dictionary.
        generate_x_0 (Callable[[Dict, ArrayLike, Dict], ArrayLike]): Function which
            generates the initial latent state given a dictionary of model parameters,
            an array of unbounded local latent variables and a data dictionary.
        forward_func (Callable[[Dict, ArrayLike, ArrayLike, Dict], ArrayLike]): Function
            which generates the next state in the latent state sequence, given a
            dictionary of model parameters, an array of unbounded local latent
            variables, the current latent state and a data dictionary.
        observation_func (Callable[[Dict, ArrayLike, ArrayLike, Dict], ArrayLike]):
            Function which generates the observation of a latent state, given a
            dictionary of model parameters, an array of unbounded local latent
            (observation noise) variables, the current latent state and a data
            dictionary.

    Returns:
        generate_from_model (
                Callable[[ArrayLike, ArrayLike, Dict], Tuple[Dict, ArrayLike]]):
            Function which given two array arguments and a data dictionary, the first
            array corresponding to all unbounded global latent variables and the second
            corresponding to all unbounded local latent variables, returns a dictionary
            of model parameters and an array corresponding to the generated latent state
            sequence.
        generate_y (
                Callable[[ArrayLike, ArrayLike, ArrayLike, Dict], ArrayLike]):
            Function which given three arrays and a data dictionary, the first array
            corresponding to all unbounded global latent variables, the second
            corresponding to all unbounded local latent variables and the third
            corresponding to all unbounded observation noise variables, returns an array
            corresponding to all observed variables.
    """

    def generate_from_model(u, v, data):
        params = generate_params(u, data)
        x_0 = generate_x_0(params, v[0], data)

        def step(x, v):
            x_ = forward_func(params, v, x, data)
            return x_, x_

        _, x_ = lax.scan(step, x_0, v[1:])
        return params, np.concatenate((x_0[None], x_))

    def generate_y(u, v, n, data):
        params, x = generate_from_model(u, v, data)
        y = api.vmap(observation_func, (None, 0, 0))(params, n, x, data)
        return y

    return generate_from_model, generate_y


class _AbstractDifferentiableGenerativeModelSystem(System):
    """Base class for constrained systems for differentiable generative models.

    Compare to in-built Mici constrained system classes, uses 'matrix-free'
    implementations of operations involving constraint function Jacobian and Gram matrix
    to allow exploiting any structure present, and also JIT compiles iterative solvers
    for projection steps to improve performance.
    """

    def __init__(
        self,
        neg_log_dens,
        grad_neg_log_dens,
        constr,
        jacob_constr_blocks,
        decompose_gram,
        lmult_by_jacob_constr,
        rmult_by_jacob_constr,
        lmult_by_inv_gram,
        lmult_by_inv_jacob_product,
        log_det_sqrt_gram,
        lmult_by_pinv_jacob_constr=None,
        normal_space_component=None,
    ):

        if lmult_by_pinv_jacob_constr is None:

            def lmult_by_pinv_jacob_constr(jacob_constr_blocks, gram_components, vct):
                return rmult_by_jacob_constr(
                    *jacob_constr_blocks,
                    lmult_by_inv_gram(*jacob_constr_blocks, *gram_components, vct,),
                )

        if normal_space_component is None:

            def normal_space_component(jacob_constr_blocks, gram_components, vct):
                return lmult_by_pinv_jacob_constr(
                    jacob_constr_blocks,
                    gram_components,
                    lmult_by_jacob_constr(*jacob_constr_blocks, vct),
                )

        def quasi_newton_projection(
            q,
            jacob_constr_blocks_prev,
            gram_components_prev,
            dt,
            constraint_tol,
            position_tol,
            divergence_tol,
            max_iters,
            norm,
        ):
            """Quasi-Newton method to solve projection onto manifold."""

            def body_func(val):
                q, mu, i, _, _ = val
                c = constr(q)
                error = norm(c)
                delta_mu = lmult_by_pinv_jacob_constr(
                    jacob_constr_blocks_prev, gram_components_prev, c
                )
                mu += delta_mu
                q -= delta_mu
                i += 1
                return q, mu, i, norm(delta_mu), error

            def cond_func(val):
                _, _, i, norm_delta_q, error = val
                diverged = np.logical_or(error > divergence_tol, np.isnan(error))
                converged = np.logical_and(
                    error < constraint_tol, norm_delta_q < position_tol
                )
                return np.logical_not(
                    np.logical_or((i >= max_iters), np.logical_or(diverged, converged))
                )

            q, mu, i, norm_delta_q, error = lax.while_loop(
                cond_func, body_func, (q, np.zeros_like(q), 0, np.inf, -1.0)
            )
            return q, mu / dt, i, norm_delta_q, error

        def newton_projection(
            q,
            jacob_constr_blocks_prev,
            dt,
            constraint_tol,
            position_tol,
            divergence_tol,
            max_iters,
            norm,
        ):
            """Newton method to solve projection onto manifold."""

            def body_func(val):
                q, mu, i, _, _ = val
                jac_blocks, c = jacob_constr_blocks(q)
                error = norm(c)
                delta_mu = rmult_by_jacob_constr(
                    *jacob_constr_blocks_prev,
                    lmult_by_inv_jacob_product(
                        *jac_blocks, *jacob_constr_blocks_prev, c
                    ),
                )
                mu += delta_mu
                q -= delta_mu
                i += 1
                return q, mu, i, norm(delta_mu), error

            def cond_func(val):
                _, _, i, norm_delta_q, error = val
                diverged = np.logical_or(error > divergence_tol, np.isnan(error))
                converged = np.logical_and(
                    error < constraint_tol, norm_delta_q < position_tol
                )
                return np.logical_not(
                    np.logical_or((i >= max_iters), np.logical_or(diverged, converged))
                )

            q, mu, i, norm_delta_q, error = lax.while_loop(
                cond_func, body_func, (q, np.zeros_like(q), 0, np.inf, -1.0)
            )
            return q, mu / dt, i, norm_delta_q, error

        self._constr = api.jit(constr)
        self._jacob_constr_blocks = api.jit(jacob_constr_blocks)
        self._decompose_gram = api.jit(decompose_gram)
        self._log_det_sqrt_gram = api.jit(log_det_sqrt_gram)
        self._val_and_grad_log_det_sqrt_gram = api.jit(
            api.value_and_grad(log_det_sqrt_gram, has_aux=True)
        )
        self._lmult_by_jacob_constr = api.jit(lmult_by_jacob_constr)
        self._rmult_by_jacob_constr = api.jit(rmult_by_jacob_constr)
        self._lmult_by_pinv_jacob_constr = api.jit(lmult_by_pinv_jacob_constr)
        self._lmult_by_inv_jacob_product = api.jit(lmult_by_inv_jacob_product)
        self._normal_space_component = api.jit(normal_space_component)
        self._quasi_newton_projection = api.jit(quasi_newton_projection, 8)
        self._newton_projection = api.jit(newton_projection, 7)
        super().__init__(neg_log_dens=neg_log_dens, grad_neg_log_dens=grad_neg_log_dens)

    def precompile_jax_functions(self, q, solver_norm=maximum_norm):
        self._neg_log_dens(q)
        self._grad_neg_log_dens(q)
        self._constr(q)
        jac_blocks, c = self._jacob_constr_blocks(q)
        gram_components = self._decompose_gram(*jac_blocks)
        self._log_det_sqrt_gram(q)
        self._val_and_grad_log_det_sqrt_gram(q)
        self._lmult_by_jacob_constr(*jac_blocks, q)
        self._rmult_by_jacob_constr(*jac_blocks, c)
        self._lmult_by_pinv_jacob_constr(jac_blocks, gram_components, c)
        self._lmult_by_inv_jacob_product(*jac_blocks, *jac_blocks, c)
        self._normal_space_component(jac_blocks, gram_components, q)
        self._quasi_newton_projection(
            q, jac_blocks, gram_components, 1, 0.1, 0.1, 10, 10, solver_norm
        )
        self._newton_projection(q, jac_blocks, 1, 0.1, 0.1, 10, 10, solver_norm)

    @cache_in_state("pos")
    def constr(self, state):
        return convert_to_numpy_pytree(self._constr(state.pos))

    @cache_in_state_with_aux("pos", "constr")
    def jacob_constr_blocks(self, state):
        return convert_to_numpy_pytree(self._jacob_constr_blocks(state.pos))

    @cache_in_state("pos")
    def gram_components(self, state):
        return convert_to_numpy_pytree(
            self._decompose_gram(*self.jacob_constr_blocks(state))
        )

    @cache_in_state_with_aux(
        "pos", ("constr", "jacob_constr_blocks", "gram_components"),
    )
    def log_det_sqrt_gram(self, state):
        val, (constr, jacob_constr_blocks, gram_components) = self._log_det_sqrt_gram(
            state.pos
        )
        return convert_to_numpy_pytree(
            (val, constr, jacob_constr_blocks, gram_components)
        )

    @cache_in_state_with_aux(
        "pos",
        ("log_det_sqrt_gram", "constr", "jacob_constr_blocks", "gram_components"),
    )
    def grad_log_det_sqrt_gram(self, state):
        (
            (val, (constr, jacob_constr_blocks, gram_components)),
            grad,
        ) = self._val_and_grad_log_det_sqrt_gram(state.pos)
        return convert_to_numpy_pytree(
            (grad, val, constr, jacob_constr_blocks, gram_components)
        )

    def h1(self, state):
        return self.neg_log_dens(state) + self.log_det_sqrt_gram(state)

    def dh1_dpos(self, state):
        return self.grad_neg_log_dens(state) + self.grad_log_det_sqrt_gram(state)

    def h2(self, state):
        return 0.5 * state.mom @ state.mom

    def dh2_dmom(self, state):
        return state.mom

    def dh2_dpos(self, state):
        return 0 * state.pos

    def dh_dpos(self, state):
        return self.dh1_dpos(state)

    def h2_flow(self, state, dt):
        state.pos += dt * self.dh2_dmom(state)

    def dh2_flow_dmom(self, dt):
        return (dt * IdentityMatrix(), IdentityMatrix())

    def normal_space_component(self, state, vct):
        return onp.asarray(
            self._normal_space_component(
                self.jacob_constr_blocks(state), self.gram_components(state), vct
            )
        )

    def lmult_by_jacob_constr(self, state, vct):
        return onp.asarray(
            self._lmult_by_jacob_constr(*self.jacob_constr_blocks(state), vct)
        )

    def rmult_by_jacob_constr(self, state, vct):
        return onp.asarray(
            self._rmult_by_jacob_constr(*self.jacob_constr_blocks(state), vct)
        )

    def lmult_by_pinv_jacob_constr(self, state, vct):
        return onp.asarray(
            self._lmult_by_pinv_jacob_constr(
                self.jacob_constr_blocks(state), self.gram_components(state), vct
            )
        )

    def lmult_by_inv_jacob_product(self, state_1, state_2, vct):
        return onp.asarray(
            self._lmult_by_inv_jacob_product(
                *self.jacob_constr_blocks(state_1),
                *self.jacob_constr_blocks(state_2),
                vct,
            )
        )

    def project_onto_cotangent_space(self, mom, state):
        mom -= self.normal_space_component(state, mom)
        return mom

    def sample_momentum(self, state, rng):
        mom = rng.standard_normal(state.pos.shape)
        mom = self.project_onto_cotangent_space(mom, state)
        return mom


class IndependentAdditiveNoiseModelSystem(_AbstractDifferentiableGenerativeModelSystem):
    """Mici system class for models with independent additive observation noise.

    Generative model is assumed to be of the form

        y = generate_y(u, n) = F(u) + s(u) * T(n)

    where `y` is a `(dim_y,)` shaped 1D array of observed variables, `u` is a `(dim_u,)`
    shaped 1D array of latent variables, `n` is a `(dim_y,)` shaped 1D array of
    observation noise variables, `F` is a differentiable function which takes a
    `(dim_u,)` shaped array as input and outputs a `(dim_y,)` shaped array, `s(u)` is a
    differentiable positive-valued function which takes a `(dim_u,)` shaped array as
    input and outputs either a scalar or a `(dim_y,)` shaped array, and `T` is a
    differentiable function which takes a `(dim_y,)` shaped array as input, outputs a
    `(dim_y,)` shaped array and acts elementwise such that it has a diagonal Jacobian.

    By default `u` and `n` are assumed to vectors of independent standard normal
    variates.

    The Jacobian of the generator function `generate_y` then has the structure

        jacobian(generate_y)(u, n) = (dy_du, diag(dy_dn)) = (
            jacobian(F)(u) + jacobian(s)(u) * T(n)[:, None], s(u) * jacobian(T)(n)
        )

    where `dy_du` is `(dim_y, dim_u)` shaped 2D array and `dy_dn` is `(dim_y,)` shaped
    1D array corresponding to the diagonal elements of the diagonal partial Jacobian
    of `generate_y` with respect to its second argument.

    If `dim_u < dim_y` then the Gram matrix `dy_du @ dy_du.T + diag(dy_dn**2)` can be
    inverted and have its determinant evaluated at `O(dim_u**2 * dim_y)` cost using
    respectively the Woodbury matrix identity and matrix determinant lemma.
    """

    def __init__(
        self,
        generate_y,
        data,
        dim_u,
        neg_log_dens=standard_normal_neg_log_dens,
        grad_neg_log_dens=standard_normal_grad_neg_log_dens,
        force_full_gram_cholesky=False,
    ):
        """
        Args:
            generate_y (callable): Differentiable function which takes three arguments
                `(u, n, data)` with `u` a length `dim_u` 1D array of latent variables,
                `n` a length `dim_y` 1D array of observation noise variables and `data`
                a dictionary of any fixed values / data used in generation, and
                outputs a length `dim_y` 1D array of generated observed variables.
            data (dict): Dictionary of fixed values used in generation of observed
                variables. Must contain an entry with key 'y_obs' corresponding to 1D
                array of shape `(dim_y,)` of observed variable values to condition on.
            dim_u (int): Dimension of latent variable array `u`.
            neg_log_dens (callable): Function which returns the negative logarithm of
                the (Lebesgue) density of the joint prior distribution on the latent
                variables `u` and observation noise variables `n`. The function should
                take a single length `dim_u + dim_y` 1D array argument corresponding to
                `concatenate((u, n))`, and return a scalar.
            grad_neg_log_dens (callable): Function which computes the gradient, or
                optionally the gradient and value, of the `neg_log_dens` function. The
                function should take a single length `dim_u + dim_y` 1D array argument
                corresponding to `concatenate((u, n))` and return either a single length
                `dim_u + dim_y` 1D array corresponding to the gradient of `neg_log_dens`
                or a 2-tuple `(grad, val)` with `grad` a length `dim_u + dim_y` 1D array
                corresponding to the gradient, and `val` the value, of `neg_log_dens`.
            force_full_gram_cholesky (bool): Boolean flag indicating whether to force
                the linear algebra operations involving the Gram matrix (i.e. solving a
                linear system in the Gram matrix or computing its log determinant) to be
                use a Cholesky of the Gram matrix rather than exploiting the Woodbury
                matrix identity and matrix determinant lemma when `dim_u < dim_y`. If
                `dim_u` is only slightly smaller than `dim_y` this may be more
                efficient.
        """

        y_obs = data["y_obs"]
        dim_y = y_obs.shape[0]

        def constr(q):
            u, n = q[:dim_u], q[dim_u:]
            return generate_y(u, n, data) - y_obs

        def jacob_constr_blocks(q):
            u, n = q[:dim_u], q[dim_u:]
            if dim_u <= dim_y:
                dy_du = api.jacfwd(generate_y)(u, n, data)
            else:
                dy_du = api.jacrev(generate_y)(u, n, data)
            y, dy_dn = api.jvp(
                lambda n: generate_y(u, n, data), (n,), (np.ones(dim_y),)
            )
            return (dy_du, dy_dn), y - y_obs

        def lmult_by_jacob_constr(dy_du, dy_dn, vct):
            vct_u, vct_n = vct[:dim_u], vct[dim_u:]
            return dy_du @ vct_u + dy_dn * vct_n

        def rmult_by_jacob_constr(dy_du, dy_dn, vct):
            return np.concatenate((vct @ dy_du, vct * dy_dn))

        if dim_u >= dim_y or force_full_gram_cholesky:

            def decompose_gram(dy_du, dy_dn):
                gram = dy_du @ dy_du.T + np.diag(dy_dn ** 2)
                chol_gram = cholesky(gram)
                return (chol_gram,)

            def lmult_by_inv_gram(dy_du, dy_dn, chol_gram, vct):
                return sla.cho_solve((chol_gram, True), vct)

            def lmult_by_inv_jacob_product(dy_du_l, dy_dn_l, dy_du_r, dy_dn_r, vct):
                jacob_product = dy_du_l @ dy_du_r.T + np.diag(dy_dn_l * dy_dn_r)
                return np.linalg.solve(jacob_product, vct)

            def log_det_sqrt_gram(q):
                (dy_du, dy_dn), c = jacob_constr_blocks(q)
                (chol_gram,) = decompose_gram(dy_du, dy_dn)
                return (
                    np.log(chol_gram.diagonal()).sum(),
                    (c, (dy_du, dy_dn), (chol_gram,)),
                )

        else:

            def decompose_gram(dy_du, dy_dn):
                dy_dn_sq = dy_dn ** 2
                cap_mtx = np.eye(dim_u) + dy_du.T @ (dy_du / dy_dn_sq[:, None])
                chol_cap_mtx = cholesky(cap_mtx)
                return chol_cap_mtx, dy_dn_sq

            def lmult_by_inv_gram(dy_du, dy_dn, chol_cap_mtx, dy_dn_sq, vct):
                return (
                    vct
                    - dy_du
                    @ sla.cho_solve((chol_cap_mtx, True), dy_du.T @ (vct / dy_dn_sq))
                ) / dy_dn_sq

            def lmult_by_inv_jacob_product(dy_du_l, dy_dn_l, dy_du_r, dy_dn_r, vct):
                dy_dn_product = dy_dn_l * dy_dn_r
                cap_mtx = np.eye(dim_u) + dy_du_r.T @ (dy_du_l / dy_dn_product[:, None])
                return (
                    vct
                    - dy_du_l
                    @ np.linalg.solve(cap_mtx, dy_du_r.T @ (vct / dy_dn_product))
                ) / dy_dn_product

            def log_det_sqrt_gram(q):
                (dy_du, dy_dn), c = jacob_constr_blocks(q)
                chol_cap_mtx, dy_dn_sq = decompose_gram(dy_du, dy_dn)
                return (
                    np.log(chol_cap_mtx.diagonal()).sum() + np.log(dy_dn_sq).sum() / 2,
                    (c, (dy_du, dy_dn), (chol_cap_mtx, dy_dn_sq)),
                )

        super().__init__(
            neg_log_dens=neg_log_dens,
            grad_neg_log_dens=grad_neg_log_dens,
            constr=constr,
            jacob_constr_blocks=jacob_constr_blocks,
            decompose_gram=decompose_gram,
            lmult_by_jacob_constr=lmult_by_jacob_constr,
            rmult_by_jacob_constr=rmult_by_jacob_constr,
            lmult_by_inv_gram=lmult_by_inv_gram,
            lmult_by_inv_jacob_product=lmult_by_inv_jacob_product,
            log_det_sqrt_gram=log_det_sqrt_gram,
        )


class HierarchicalLatentVariableModelSystem(
    _AbstractDifferentiableGenerativeModelSystem
):
    """Mici system class for hierarchical latent variable models."""

    def __init__(
        self,
        generate_y,
        data,
        dim_u,
        neg_log_dens=standard_normal_neg_log_dens,
        grad_neg_log_dens=standard_normal_grad_neg_log_dens,
    ):
        """
        Args:
            data (dict): Dictionary of fixed values used in generating the observed
                variables. Must also contain an entry with key 'y_obs' corresponding to
                1D array of shape `(dim_y,)` of observed variable values to condition on
            dim_u (int): Dimension of latent variable array `u`.
            neg_log_dens (callable): Function which returns the negative logarithm of
                the (Lebesgue) density of the joint prior distribution on the global
                latent variables `u`, local latent variables `v` and observation noise
                variables `n`. The function should take a single length `dim_u + 2 *
                dim_y` 1D array argument corresponding to `concatenate((u, v, n))` and
                return a scalar.
            grad_neg_log_dens (callable): Function which computes the gradient, or
                optionally the gradient and value, of the `neg_log_dens` function. The
                function should take a single length `dim_u + 2 * dim_y` 1D array
                argument corresponding to `concatenate((u, v, n))` and return either a
                single length `dim_u + 2 * dim_y` 1D array corresponding to the gradient
                of `neg_log_dens` or a 2-tuple `(grad, val)` with `grad` a length `dim_u
                + 2 * dim_y` 1D array corresponding to the gradient, and `val` the
                value, of `neg_log_dens`.
        """

        dim_y = data["y_obs"].shape[0]

        def constr(q):
            u, v, n = q[:dim_u], q[dim_u : dim_u + dim_y], q[dim_u + dim_y :]
            return generate_y(u, v, n, data) - data["y_obs"]

        def jacob_constr_blocks(q):
            u, v, n = q[:dim_u], q[dim_u : dim_u + dim_y], q[dim_u + dim_y :]
            if dim_u <= dim_y:
                dy_du = api.jacfwd(generate_y)(u, v, n, data)
            else:
                dy_du = api.jacrev(generate_y)(u, v, n, data)
            y, dy_dv = api.jvp(
                lambda v: generate_y(u, v, n, data), (v,), (np.ones(dim_y),)
            )
            y, dy_dn = api.jvp(
                lambda n: generate_y(u, v, n, data), (n,), (np.ones(dim_y),)
            )
            return (dy_du, dy_dv, dy_dn), y - data["y_obs"]

        def lmult_by_jacob_constr(dy_du, dy_dv, dy_dn, vct):
            vct_u, vct_v, vct_n = (
                vct[:dim_u],
                vct[dim_u : dim_u + dim_y],
                vct[dim_u + dim_y :],
            )
            return dy_du @ vct_u + dy_dv * vct_v + dy_dn * vct_n

        def rmult_by_jacob_constr(dy_du, dy_dv, dy_dn, vct):
            return np.concatenate((vct @ dy_du, vct * dy_dv, vct * dy_dn))

        if dim_u >= dim_y:

            def decompose_gram(dy_du, dy_dv, dy_dn):
                gram = dy_du @ dy_du.T + np.diag(dy_dv ** 2 + dy_dn ** 2)
                chol_gram = cholesky(gram)
                return (chol_gram,)

            def lmult_by_inv_gram(dy_du, dy_dv, dy_dn, chol_gram, vct):
                return sla.cho_solve((chol_gram, True), vct)

            def lmult_by_inv_jacob_product(
                dy_du_l, dy_dv_l, dy_dn_l, dy_du_r, dy_dv_r, dy_dn_r, vct
            ):
                jacob_product = dy_du_l @ dy_du_r.T + np.diag(
                    dy_dv_l * dy_dv_r + dy_dn_l * dy_dn_r
                )
                return np.linalg.solve(jacob_product, vct)

            def log_det_sqrt_gram(q):
                (dy_du, dy_dv, dy_dn), c = jacob_constr_blocks(q)
                (chol_gram,) = decompose_gram(dy_du, dy_dv, dy_dn)
                return (
                    np.log(chol_gram.diagonal()).sum(),
                    (c, (dy_du, dy_dv, dy_dn), (chol_gram,)),
                )

        else:

            def decompose_gram(dy_du, dy_dv, dy_dn):
                dy_dv_sq_plus_dy_dn_sq = dy_dv ** 2 + dy_dn ** 2
                cap_mtx = np.eye(dim_u) + dy_du.T @ (
                    dy_du / dy_dv_sq_plus_dy_dn_sq[:, None]
                )
                chol_cap_mtx = cholesky(cap_mtx)
                return chol_cap_mtx, dy_dv_sq_plus_dy_dn_sq

            def lmult_by_inv_gram(
                dy_du, dy_dv, dy_dn, chol_cap_mtx, dy_dv_sq_plus_dy_dn_sq, vct
            ):
                return (
                    vct
                    - dy_du
                    @ sla.cho_solve(
                        (chol_cap_mtx, True), dy_du.T @ (vct / dy_dv_sq_plus_dy_dn_sq)
                    )
                ) / dy_dv_sq_plus_dy_dn_sq

            def lmult_by_inv_jacob_product(
                dy_du_l, dy_dv_l, dy_dn_l, dy_du_r, dy_dv_r, dy_dn_r, vct
            ):
                dy_dv_plus_dy_dn_product = dy_dv_l * dy_dv_r + dy_dn_l * dy_dn_r
                cap_mtx = np.eye(dim_u) + dy_du_r.T @ (
                    dy_du_l / dy_dv_plus_dy_dn_product[:, None]
                )
                return (
                    vct
                    - dy_du_l
                    @ np.linalg.solve(
                        cap_mtx, dy_du_r.T @ (vct / dy_dv_plus_dy_dn_product)
                    )
                ) / dy_dv_plus_dy_dn_product

            def log_det_sqrt_gram(q):
                (dy_du, dy_dv, dy_dn), c = jacob_constr_blocks(q)
                chol_cap_mtx, dy_dv_sq_plus_dy_dn_sq = decompose_gram(
                    dy_du, dy_dv, dy_dn
                )
                return (
                    np.log(chol_cap_mtx.diagonal()).sum()
                    + np.log(dy_dv_sq_plus_dy_dn_sq).sum() / 2,
                    (c, (dy_du, dy_dv, dy_dn), (chol_cap_mtx, dy_dv_sq_plus_dy_dn_sq)),
                )

        super().__init__(
            neg_log_dens=neg_log_dens,
            grad_neg_log_dens=grad_neg_log_dens,
            constr=constr,
            jacob_constr_blocks=jacob_constr_blocks,
            decompose_gram=decompose_gram,
            lmult_by_jacob_constr=lmult_by_jacob_constr,
            rmult_by_jacob_constr=rmult_by_jacob_constr,
            lmult_by_inv_gram=lmult_by_inv_gram,
            lmult_by_inv_jacob_product=lmult_by_inv_jacob_product,
            log_det_sqrt_gram=log_det_sqrt_gram,
        )


class GaussianProcessModelSystem(_AbstractDifferentiableGenerativeModelSystem):
    """Mici system class for Gaussian process models.

    Generative model is assumed to be of the form

        y = generate_y(u, n, data) = cholesky(covar_func(u, data)) @ n

    where `y` is a `(dim_y,)` shaped 1D array of observed variables, `u` is a `(dim_u,)`
    shaped 1D array of global latent variables, `n` is a `(dim_y,)` shaped 1D array of
    local latent variables and `data` is a dictionary of fixed values used in generation
    of observed variables (e.g. inputs associated with each observed output).

    It is assumed `covar_func` is a differentiable function which takes two arguments, a
    length `dim_u` 1D array of latent variables and a data dictionary, and outputs a
    `(dim_y, dim_y)` 2D array corresponding to a positive-definite matrix.
    """

    def __init__(
        self,
        covar_func,
        data,
        dim_u,
        neg_log_dens=standard_normal_neg_log_dens,
        grad_neg_log_dens=standard_normal_grad_neg_log_dens,
    ):
        """
        Args:
            covar_func (callable): Differentiable function which takes two arguments, a
                length `dim_u` 1D array of latent variables and a data dictionary, and
                outputs a `(dim_y, dim_y)` 2D array corresponding to a positive-definite
                matrix.
            data (dict): Dictionary of fixed values used in constructing covariance
                matrix. Must also contain an entry with key 'y_obs' corresponding to 1D
                array of shape `(dim_y,)` of observed variable values to condition on.
            dim_u (int): Dimension of latent variable array `u`.
            neg_log_dens (callable): Function which returns the negative logarithm of
                the (Lebesgue) density of the joint prior distribution on the latent
                variables `u` and observation noise variables `n`. The function should
                take a single length `dim_u + dim_y` 1D array argument corresponding to
                `concatenate((u, n))`, and return a scalar.
            grad_neg_log_dens (callable): Function which computes the gradient, or
                optionally the gradient and value, of the `neg_log_dens` function. The
                function should take a single length `dim_u + dim_y` 1D array argument
                corresponding to `concatenate((u, n))` and return either a single length
                `dim_u + dim_y` 1D array corresponding to the gradient of `neg_log_dens`
                or a 2-tuple `(grad, val)` with `grad` a length `dim_u + dim_y` 1D array
                corresponding to the gradient, and `val` the value, of `neg_log_dens`.
        """

        def constr(q):
            u, n = q[:dim_u], q[dim_u:]
            covar = covar_func(u, data)
            return cholesky(covar) @ n - data["y_obs"]

        def jacob_constr_blocks(q):
            u, n = q[:dim_u], q[dim_u:]
            covar, dcovar_du = api.vmap(
                api.partial(api.jvp, lambda u: covar_func(u, data), (u,)),
                out_axes=(None, 1),
            )((np.identity(dim_u),))
            chol_covar = cholesky(covar)
            dy_du = api.vmap(jvp_cholesky_mtx_mult_by_vct, (1, None, None), 1)(
                dcovar_du, chol_covar, n
            )
            return (dy_du, chol_covar), chol_covar @ n - data["y_obs"]

        def lmult_by_jacob_constr(dy_du, dy_dn, vct):
            vct_u, vct_n = vct[:dim_u], vct[dim_u:]
            return dy_du @ vct_u + dy_dn @ vct_n

        def rmult_by_jacob_constr(dy_du, dy_dn, vct):
            return np.concatenate((vct @ dy_du, vct @ dy_dn))

        def decompose_gram(dy_du, dy_dn):
            cap_mtx = np.eye(dim_u) + dy_du.T @ sla.cho_solve((dy_dn, True), dy_du)
            chol_cap_mtx = cholesky(cap_mtx)
            return (chol_cap_mtx,)

        def lmult_by_inv_gram(dy_du, dy_dn, chol_cap_mtx, vct):
            return sla.cho_solve(
                (dy_dn, True),
                vct
                - dy_du
                @ sla.cho_solve(
                    (chol_cap_mtx, True), dy_du.T @ sla.cho_solve((dy_dn, True), vct)
                ),
            )

        def lmult_by_inv_jacob_product(dy_du_l, dy_dn_l, dy_du_r, dy_dn_r, vct):
            cap_mtx = np.eye(dim_u) + dy_du_r.T @ lu_triangular_solve(
                dy_dn_l, dy_dn_r.T, dy_du_l
            )
            return lu_triangular_solve(
                dy_dn_l,
                dy_dn_r.T,
                vct
                - dy_du_l
                @ np.linalg.solve(
                    cap_mtx, dy_du_r.T @ lu_triangular_solve(dy_dn_l, dy_dn_r.T, vct),
                ),
            )

        def log_det_sqrt_gram(q):
            (dy_du, dy_dn), c = jacob_constr_blocks(q)
            (chol_cap_mtx,) = decompose_gram(dy_du, dy_dn)
            return (
                np.log(chol_cap_mtx.diagonal()).sum() + np.log(dy_dn.diagonal()).sum(),
                (c, (dy_du, dy_dn), (chol_cap_mtx,)),
            )

        super().__init__(
            neg_log_dens=neg_log_dens,
            grad_neg_log_dens=grad_neg_log_dens,
            constr=constr,
            jacob_constr_blocks=jacob_constr_blocks,
            decompose_gram=decompose_gram,
            lmult_by_jacob_constr=lmult_by_jacob_constr,
            rmult_by_jacob_constr=rmult_by_jacob_constr,
            lmult_by_inv_gram=lmult_by_inv_gram,
            lmult_by_inv_jacob_product=lmult_by_inv_jacob_product,
            log_det_sqrt_gram=log_det_sqrt_gram,
        )


class GeneralGaussianProcessModelSystem(_AbstractDifferentiableGenerativeModelSystem):
    """Mici system class for Gaussian process models with general observation noise.

    Generative model is assumed to be of the form

        y = (
            cholesky(covar_func(u, data)) @ v +
            noise_scale_func(u, data) * noise_transform_func(n)
        )

    where `y` is a `(dim_y,)` shaped 1D array of observed variables, `u` is a `(dim_u,)`
    shaped 1D array of global latent variables, `data` is a dictionary of fixed values
    used in constructing the covariance matrix, `v` is a `(dim_y,)` shaped 1D array of
    local latent variables and `n` is a `(dim_y,)` shaped 1D array of observation noise
    variables.

    It is assumed `covar_func` is a differentiable function which takes two arguments, a
    length `dim_u` 1D array of latent variables and a data dictionary, and outputs a
    `(dim_y, dim_y)` 2D array corresponding to a positive-definite matrix,
    `noise_scale_func` is a differentiable positive-valued function which takes a
    `(dim_u,)` shaped array and data dictionary as input and outputs a scalar, and
    `noise_transform_func` is a differentiable function which takes a `(dim_y,)` shaped
    array as input, outputs a `(dim_y,)` shaped array and acts elementwise such that it
    has a diagonal Jacobian.
    """

    def __init__(
        self,
        covar_func,
        noise_scale_func,
        noise_transform_func,
        data,
        dim_u,
        neg_log_dens=standard_normal_neg_log_dens,
        grad_neg_log_dens=standard_normal_grad_neg_log_dens,
    ):
        """
        Args:
            covar_func (callable): Differentiable function which takes two arguments, a
                length `dim_u` 1D array of latent variables and a data dictionary, and
                outputs a `(dim_y, dim_y)` 2D array corresponding to a positive-definite
                matrix.
            noise_scale_func (callable): Differentiable function which takes two
                arguments, a length `dim_u` 1D array of latent variables and a data
                dictionary and outputs a positive scalar.
            noise_transform_func (callable): Differentiable function which takes a
                single argument, a length `dim_y` 1D array and outputs a `(dim_y,)` 1D
                array, with the function acting elementwise such that it has a diagonal
                Jacobian. If `None` assumed to be the identity function.
            data (dict): Dictionary of fixed values used in the model. Must also contain
                an entry with key 'y_obs' corresponding to 1D array of shape `(dim_y,)`
                of observed variable values to condition on.
            dim_u (int): Dimension of latent variable array `u`.
            neg_log_dens (callable): Function which returns the negative logarithm of
                the (Lebesgue) density of the joint prior distribution on the global
                latent variables `u`, local latent variables `v` and observation noise
                variables `n`. The function should take a single length `dim_u + 2 *
                dim_y` 1D array argument corresponding to `concatenate((u, v, n))` and
                return a scalar.
            grad_neg_log_dens (callable): Function which computes the gradient, or
                optionally the gradient and value, of the `neg_log_dens` function. The
                function should take a single length `dim_u + 2 * dim_y` 1D array
                argument corresponding to `concatenate((u, v, n))` and return either a
                single length `dim_u + 2 * dim_y` 1D array corresponding to the gradient
                of `neg_log_dens` or a 2-tuple `(grad, val)` with `grad` a length `dim_u
                + 2 * dim_y` 1D array corresponding to the gradient, and `val` the
                value, of `neg_log_dens`.
        """

        dim_y = data["y_obs"].shape[0]

        def generate_y(u, v, n):
            covar = covar_func(u, data)
            t = noise_transform_func(n) if noise_transform_func is not None else n
            return cholesky(covar) @ v + noise_scale_func(u, data) * t

        def constr(q):
            u, v, n = q[:dim_u], q[dim_u : dim_u + dim_y], q[dim_u + dim_y :]
            return generate_y(u, v, n) - data["y_obs"]

        def jacob_constr_blocks(q):
            u, v, n = q[:dim_u], q[dim_u : dim_u + dim_y], q[dim_u + dim_y :]
            covar, dcovar_du = api.vmap(
                api.partial(api.jvp, lambda u: covar_func(u, data), (u,)),
                out_axes=(None, 1),
            )((np.identity(dim_u),))
            chol_covar = cholesky(covar)
            s, ds_du = api.value_and_grad(noise_scale_func)(u, data)
            if noise_transform_func is not None:
                t, dt_dn = api.jvp(noise_transform_func, (n,), (np.ones(dim_y),))
            else:
                t, dt_dn = n, np.ones(dim_y)
            dy_du = ds_du[None, :] * t[:, None] + api.vmap(
                jvp_cholesky_mtx_mult_by_vct, (1, None, None), 1
            )(dcovar_du, chol_covar, v)
            return (
                (dy_du, chol_covar, s * dt_dn, covar),
                chol_covar @ v + s * t - data["y_obs"],
            )

        def lmult_by_jacob_constr(dy_du, dy_dv, dy_dn, covar, vct):
            vct_u, vct_v, vct_n = (
                vct[:dim_u],
                vct[dim_u : dim_u + dim_y],
                vct[dim_u + dim_y :],
            )
            return dy_du @ vct_u + dy_dv @ vct_v + dy_dn * vct_n

        def rmult_by_jacob_constr(dy_du, dy_dv, dy_dn, covar, vct):
            return np.concatenate((vct @ dy_du, vct @ dy_dv, vct * dy_dn))

        def decompose_gram(dy_du, dy_dv, dy_dn, covar):
            # covar = chol_covar @ chol_covar.T = dy_dv @ dy_dv.T
            gram = dy_du @ dy_du.T + covar + np.diag(dy_dn ** 2)
            chol_gram = cholesky(gram)
            return (chol_gram,)

        def lmult_by_inv_gram(dy_du, dy_dv, dy_dn, covar, chol_gram, vct):
            return sla.cho_solve((chol_gram, True), vct)

        def lmult_by_inv_jacob_product(
            dy_du_l, dy_dv_l, dy_dn_l, covar_l, dy_du_r, dy_dv_r, dy_dn_r, covar_r, vct
        ):
            jacob_product = (
                dy_du_l @ dy_du_r.T + dy_dv_l @ dy_dv_r.T + np.diag(dy_dn_l * dy_dn_r)
            )
            return sla.solve(jacob_product, vct)

        def log_det_sqrt_gram(q):
            (dy_du, dy_dv, dy_dn, covar), c = jacob_constr_blocks(q)
            (chol_gram,) = decompose_gram(dy_du, dy_dv, dy_dn, covar)
            return (
                np.log(chol_gram.diagonal()).sum(),
                (c, (dy_du, dy_dv, dy_dn, covar), (chol_gram,)),
            )

        super().__init__(
            neg_log_dens=neg_log_dens,
            grad_neg_log_dens=grad_neg_log_dens,
            constr=constr,
            jacob_constr_blocks=jacob_constr_blocks,
            decompose_gram=decompose_gram,
            lmult_by_jacob_constr=lmult_by_jacob_constr,
            rmult_by_jacob_constr=rmult_by_jacob_constr,
            lmult_by_inv_gram=lmult_by_inv_gram,
            lmult_by_inv_jacob_product=lmult_by_inv_jacob_product,
            log_det_sqrt_gram=log_det_sqrt_gram,
        )


class PartiallyInvertibleStateSpaceModelSystem(
    _AbstractDifferentiableGenerativeModelSystem
):
    """System class for scalar state space models with invertible observation functions.

    Generative model is assumed to be of the form

        params = generate_params(u, data)
        x[0] = generate_x_0(params, v[0], data)
        for t in range(dim_y):
            x[t] = forward_func(params, v[t], x[t - 1], data)
            y[t] = observation_func(params, n[t], x[t], data)

    If `inverse_observation_func` corresponds to the inverse of `observation_func` in
    its third argument,

        observation_func(
            params, n, inverse_observation_func(params, n, y, data), data) == y

    then we can define a 'split' constraint function for the generative model as follows

        def constr_split(u, v, n, y, data):
            params = generate_params(u, data)
            x = [
                inverse_observation_func(params, n[t], y[t], data)
                for t in range(1, dim_y)
            ]
            return array(
                [generate_x_0(params, v[0], data) - x[0]] +
                [
                    forward_func(params, v[t], x[t-1], data) - x[t]
                    for t in range(1, dim_y)
                ]
            ), x

    where `y` is a `(dim_y,)` shaped 1D array of observed variables, `u` is a `(dim_u,)`
    shaped 1D array of global latent variables, `v` is a `(dim_y,)` shaped 1D array of
    local latent variables, `n` is a `(dim_y,)` shaped 1D array of observation noise
    variables and `data` is a dictionary of fixed values / data used by model.
    """

    def __init__(
        self,
        constr_split,
        jacob_constr_split_blocks,
        data,
        dim_u,
        neg_log_dens=standard_normal_neg_log_dens,
        grad_neg_log_dens=standard_normal_grad_neg_log_dens,
    ):
        dim_y = data["y_obs"].shape[0]

        if jacob_constr_split_blocks is None:

            def jacob_constr_split_blocks(u, v, n, y, data):
                dc_du = api.jacfwd(lambda u_: constr_split(u_, v, n, y, data)[0])(u)
                one_vct = np.ones(dim_y)
                alt_vct = (-1.0) ** np.arange(dim_y)
                _, dx_dy = api.jvp(
                    lambda y_: constr_split(u, v, n, y_, data)[1], (y,), (one_vct,)
                )
                c, dc_dv = api.jvp(
                    lambda v_: constr_split(u, v_, n, y, data)[0], (v,), (one_vct,)
                )
                _, dc_dn_1 = api.jvp(
                    lambda n_: constr_split(u, v, n_, y, data)[0], (n,), (one_vct,)
                )
                _, dc_dn_a = api.jvp(
                    lambda n_: constr_split(u, v, n_, y, data)[0], (n,), (alt_vct,)
                )
                dc_dn = (
                    (dc_dn_1 + dc_dn_a * alt_vct) / 2,
                    (dc_dn_1[1:] - dc_dn_a[1:] * alt_vct[1:]) / 2,
                )
                return (dc_du, dc_dv, dc_dn, dx_dy), c

        def constr(q):
            u, v, n = np.split(q, (dim_u, dim_u + dim_y))
            return constr_split(u, v, n, data["y_obs"], data)[0]

        def jacob_constr_blocks(q):
            u, v, n = np.split(q, (dim_u, dim_u + dim_y))
            return jacob_constr_split_blocks(u, v, n, data["y_obs"], data)

        def lmult_by_jacob_constr(dc_du, dc_dv, dc_dn, dx_dy, vct):
            vct_u, vct_v, vct_n = (
                vct[:dim_u],
                vct[dim_u : dim_u + dim_y],
                vct[dim_u + dim_y :],
            )
            return (
                dc_du @ vct_u
                + dc_dv * vct_v
                + dc_dn[0] * vct_n
                + np.pad(dc_dn[1] * vct_n[:-1], (1, 0))
            )

        def rmult_by_jacob_constr(dc_du, dc_dv, dc_dn, dx_dy, vct):
            return np.concatenate(
                (
                    vct @ dc_du,
                    vct * dc_dv,
                    vct * dc_dn[0] + np.pad(dc_dn[1] * vct[1:], (0, 1)),
                )
            )

        def decompose_gram(dc_du, dc_dv, dc_dn, dx_dy):
            a = dc_dn[0][:-1] * dc_dn[1]
            b = dc_dv ** 2 + dc_dn[0] ** 2 + np.pad(dc_dn[1] ** 2, (1, 0))
            cap_mtx = np.eye(dim_u) + dc_du.T @ api.vmap(
                tridiagonal_solve, (None, None, None, 1), 1
            )(a, b, a, dc_du)
            chol_cap_mtx = cholesky(cap_mtx)
            return (a, b, chol_cap_mtx)

        def lmult_by_inv_gram(dc_du, dc_dv, dc_dn, dx_dy, a, b, chol_cap_mtx, vct):
            return tridiagonal_solve(
                a,
                b,
                a,
                vct
                - dc_du
                @ sla.cho_solve(
                    (chol_cap_mtx, True), dc_du.T @ tridiagonal_solve(a, b, a, vct)
                ),
            )

        def lmult_by_inv_jacob_product(
            dc_du_l, dc_dv_l, dc_dn_l, dx_dy_l, dc_du_r, dc_dv_r, dc_dn_r, dx_dy_r, vct
        ):
            a = dc_dn_l[1] * dc_dn_r[0][:-1]
            b = (
                dc_dv_l * dc_dv_r
                + dc_dn_l[0] * dc_dn_r[0]
                + np.pad(dc_dn_l[1] * dc_dn_r[1], (1, 0))
            )
            c = dc_dn_l[0][:-1] * dc_dn_r[1]
            cap_mtx = np.eye(dim_u) + dc_du_r.T @ api.vmap(
                tridiagonal_solve, (None, None, None, 1), 1
            )(a, b, c, dc_du_l)
            return tridiagonal_solve(
                a,
                b,
                c,
                vct
                - dc_du_l
                @ np.linalg.solve(cap_mtx, dc_du_r.T @ tridiagonal_solve(a, b, c, vct)),
            )

        def log_det_sqrt_gram(q):
            (dc_du, dc_dv, dc_dn, dx_dy), c = jacob_constr_blocks(q)
            (a, b, chol_cap_mtx,) = decompose_gram(dc_du, dc_dv, dc_dn, dx_dy)
            return (
                np.log(chol_cap_mtx.diagonal()).sum()
                + tridiagonal_pos_def_log_det(a, b) / 2
                - np.log(abs(dx_dy)).sum(),
                (c, (dc_du, dc_dv, dc_dn, dx_dy), (a, b, chol_cap_mtx,)),
            )

        super().__init__(
            neg_log_dens=neg_log_dens,
            grad_neg_log_dens=grad_neg_log_dens,
            constr=constr,
            jacob_constr_blocks=jacob_constr_blocks,
            decompose_gram=decompose_gram,
            lmult_by_jacob_constr=lmult_by_jacob_constr,
            rmult_by_jacob_constr=rmult_by_jacob_constr,
            lmult_by_inv_gram=lmult_by_inv_gram,
            lmult_by_inv_jacob_product=lmult_by_inv_jacob_product,
            log_det_sqrt_gram=log_det_sqrt_gram,
        )


class AutoPartiallyInvertibleStateSpaceModelSystem(
    PartiallyInvertibleStateSpaceModelSystem
):
    """System class for scalar state space models with invertible observation functions.

    Generative model is assumed to be of the form

        params = generate_params(u, data)
        x[0] = generate_x_0(params, v[0], data)
        for t in range(dim_y):
            x[t] = forward_func(params, v[t], x[t - 1], data)
            y[t] = observation_func(params, n[t], x[t], data)

    If `inverse_observation_func` corresponds to the inverse of `observation_func` in
    its third argument,

        observation_func(
            params, n, inverse_observation_func(params, n, y, data), data) == y

    then we can define a 'split' constraint function for the generative model as follows

        def constr_split(u, v, n, y, data):
            params = generate_params(u, data)
            x = [
                inverse_observation_func(params, n[t], y[t], data)
                for t in range(1, dim_y)
            ]
            return array(
                [generate_x_0(params, v[0], data) - x[0]] +
                [
                    forward_func(params, v[t], x[t-1], data) - x[t]
                    for t in range(1, dim_y)
                ]
            ), x

    where `y` is a `(dim_y,)` shaped 1D array of observed variables, `u` is a `(dim_u,)`
    shaped 1D array of global latent variables, `v` is a `(dim_y,)` shaped 1D array of
    local latent variables, `n` is a `(dim_y,)` shaped 1D array of observation noise
    variables and `data` is a dictionary of fixed values / data used by model.

    Compared to the `PartiallyInvertibleStateSpaceModelSystem` class this class
    automatically constructs the required 'split' constraint function (and function to
    evaluate the non-zero blocks of its Jacobian) using JAX operators from functions
    `generate_x_0`, `forward_func` and `inverse_observation_func`. This simplifies
    constructing a system object but potentially gives a performance hit compared to
    manually constructing these functions (both at compile and run time).
    """

    def __init__(
        self,
        generate_params,
        generate_x_0,
        forward_func,
        inverse_observation_func,
        data,
        dim_u,
        neg_log_dens=standard_normal_neg_log_dens,
        grad_neg_log_dens=standard_normal_grad_neg_log_dens,
    ):
        def _generate_x_0(u, v_0):
            return generate_x_0(generate_params(u, data), v_0, data)

        def _forward_func(u, v, x):
            return forward_func(generate_params(u, data), v, x, data)

        def _inverse_observation_func(u, n, y):
            return inverse_observation_func(generate_params(u, data), n, y, data)

        def constr_split(u, v, n, y, data):
            x = api.vmap(_inverse_observation_func, (None, 0, 0))(u, n, y)
            return (
                np.concatenate(
                    (
                        (_generate_x_0(u, v[0]) - x[0])[None],
                        api.vmap(_forward_func, (None, 0, 0))(u, v[1:], x[:-1]) - x[1:],
                    )
                ),
                x,
            )

        def jacob_constr_split_blocks(u, v, n, y, data):
            x, (dx_du, dx_dn, dx_dy) = api.vmap(
                api.value_and_grad(_inverse_observation_func, (0, 1, 2)), (None, 0, 0)
            )(u, n, y)
            x0, (dx0_du, dx0_dv0) = api.value_and_grad(_generate_x_0, (0, 1))(u, v[0])
            xp, (dxp_du, dxp_dvp, dxp_dxm) = api.vmap(
                api.value_and_grad(_forward_func, (0, 1, 2)), (None, 0, 0)
            )(u, v[1:], x[:-1])
            c = np.concatenate((np.atleast_1d(x0 - x[0]), xp - x[1:]))
            dc_du = np.concatenate(
                (
                    (dx0_du - dx_du[0])[None],
                    dxp_du + dxp_dxm[:, None] * dx_du[:-1] - dx_du[1:],
                ),
                0,
            )
            dc_dv = np.concatenate(((dx0_dv0)[None], dxp_dvp))
            dc_dn = (-dx_dn, dxp_dxm * dx_dn[:-1])
            return (dc_du, dc_dv, dc_dn, dx_dy), c

        super().__init__(
            constr_split=constr_split,
            jacob_constr_split_blocks=jacob_constr_split_blocks,
            data=data,
            dim_u=dim_u,
            neg_log_dens=neg_log_dens,
            grad_neg_log_dens=grad_neg_log_dens,
        )

