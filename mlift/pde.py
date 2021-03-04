import numpy as np
import fenics
import ufl
from scipy.special import gamma
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from sksparse.cholmod import cholesky as sparse_cholesky
import mici


def apply_boundary_conditions(obj, boundary_conditions):
    for boundary_condition in boundary_conditions:
        boundary_condition.apply(obj)


def construct_solver(bilinear_form, boundary_conditions):
    matrix = fenics.assemble(bilinear_form)
    apply_boundary_conditions(matrix, boundary_conditions)
    return fenics.LUSolver(matrix)


def solve_with_boundary_conditions(
    solver, boundary_conditions, b_vector, x_func_space_or_x
):
    x = (
        fenics.Function(x_func_space_or_x)
        if isinstance(x_func_space_or_x, fenics.FunctionSpace)
        else x_func_space_or_x
    )
    apply_boundary_conditions(b_vector, boundary_conditions)
    solver.solve(x.vector(), b_vector)
    return x


def fenics_matrix_to_csr_matrix(fenics_matrix):
    return csr_matrix(
        fenics.as_backend_type(fenics_matrix).mat().getValuesCSR()[::-1],
        shape=(fenics_matrix.size(0), fenics_matrix.size(1)),
    )


def fenics_matrix_to_linear_operator(fenics_matrix):
    return aslinearoperator(fenics_matrix_to_csr_matrix(fenics_matrix))


def psd_fenics_matrix_to_cholmod_factor_and_shape(fenics_matrix):
    return (
        sparse_cholesky(fenics_matrix_to_csr_matrix(fenics_matrix).tocsc()),
        (fenics_matrix.size(0), fenics_matrix.size(1)),
    )


def cholmod_factor_to_inv_linear_operator(cholmod_factor, shape):
    return LinearOperator(
        shape=shape,
        matvec=cholmod_factor,  ## cholmod_factor(x) == cholmod_factor.solve_A(x)
        rmatvec=cholmod_factor,  ## matrix is symmetric
    )


def cholmod_factor_to_sqrt_linear_operator(cholmod_factor, shape):
    L = cholmod_factor.L()

    def lmult_by_sqrt(x):
        return cholmod_factor.apply_Pt(L @ x)

    def rmult_by_sqrt(x):
        return cholmod_factor.apply_P(x) @ L

    return LinearOperator(shape=shape, matvec=lmult_by_sqrt, rmatvec=rmult_by_sqrt)


def cholmod_factor_to_sqrt_inv_linear_operator(cholmod_factor, shape):
    def lmult_by_sqrt_inv(x):
        return cholmod_factor.apply_Pt(cholmod_factor.solve_Lt(x, False))

    def rmult_by_sqrt_inv(x):
        return cholmod_factor.solve_L(cholmod_factor.apply_P(x), False)

    return LinearOperator(
        shape=shape, matvec=lmult_by_sqrt_inv, rmatvec=rmult_by_sqrt_inv
    )

def construct_mesh_and_observation_coordinates(
    mesh_type, mesh_resolution, obs_resolution
):
    if mesh_type == "square":
        mesh = fenics.UnitSquareMesh(mesh_resolution, mesh_resolution)
        observation_coordinates = np.stack(
            np.meshgrid(
                np.linspace(0, 1, 2 * obs_resolution + 1)[1:-1:2],
                np.linspace(0, 1, 2 * obs_resolution + 1)[1:-1:2],
            ),
            -1,
        ).reshape((-1, 2))
    elif mesh_type == "circle":
        mesh = fenics.UnitDiscMesh.create(fenics.MPI.comm_world, mesh_resolution, 1, 2)
        observation_coordinates = []
        for i, radius in enumerate(np.linspace(0, 1, obs_resolution, False)):
            angles = np.linspace(0, 2 * np.pi, max(1, i * obs_resolution), False)
            observation_coordinates.append(
                np.stack([np.sin(angles) * radius, np.cos(angles) * radius], -1)
            )
        observation_coordinates = np.concatenate(observation_coordinates, 0)
    else:
        raise ValueError(f"Unknown mesh type {mesh_type}")
    return mesh, observation_coordinates


def construct_matern_covar_sqrt_operator(
    func_space, length_scale=1, amplitude=1, order=1,
):
    """Construct covariance matrix square-root of discretised Matern Gaussian field.

    Creates a `LinearOperator` object corresponding a non-symmetric matrix square-root
    (i.e. a matrix `S` such that `S @ S.T == C` for a given positive-definite matrix
    `C`) of the covariance matrix of a finite element method discretisation of a
    stochastic partial differential equation (SPDE) based formulation of a Matern
    covariance Gaussian field.

    The Matern covariance function is defined as

        def covar_func(x, y):
            kappa = (8 * order)**0.5 / length_scale
            r = sqrt(((x - y)**2).sum())
            return (
                amplitude**2 / (2**(order - 1) * gamma(order)) *
                (kappa * r)**order * kv(order, kappa * r)
            )

    where `kv` is the modified Bessel function of the second kind and `gamma` the
    Gamma function.

    The 'natural' Neumann boundary conditions specifying a zero normal derivative are
    implicitly assumed.

    Args:
        function_space (ufl.FunctionSpace): Unified form language object defining
            finite element function space SPDE is discretized on.
        length_scale (float): Positive scalar defining length-scale parameter of Matern
            covariance function.
        amplitude (float): Positive scalar defining amplitude parameter (marignal
            standard deviation) of Matern covariance function.
        order (float): Positive scalar defining order parameter of Matern covariance
            function. Must be integer for even dimensions and half-integer for odd
            dimensions.

    Returns:
        LinearOperator: 'Matrix-like' object representing the covariance square-root.


    References:

    Lindgren, F., Rue, H., & Lindström, J. (2011). An explicit link between Gaussian
    fields and Gaussian Markov random fields: the stochastic partial differential
    equation approach.  Journal of the Royal Statistical Society:
    Series B (Statistical Methodology), 73(4), 423-498.

    Croci, M., Giles, M. B., Rognes, M. E., & Farrell, P. E. (2018). Efficient white
    noise sampling and coupling for multilevel Monte Carlo with nonnested meshes.
    SIAM/ASA Journal on Uncertainty Quantification, 6(4), 1630-1655.
    """

    dimension = func_space.mesh().topology().dim()
    alpha = order + dimension / 2
    if alpha < 1 or int(alpha) != alpha:
        raise ValueError(
            "Implemented only for when `order + dimension / 2` is a positive integer "
            "where `dimension` is the spatial dimension of the mesh elements"
        )
    alpha = int(alpha)
    kappa = (8 * order) ** 0.5 / length_scale
    eta = (
        amplitude
        * (
            (gamma(alpha) / gamma(order))
            * length_scale ** dimension
            * (np.pi / (2 * order)) ** (dimension / 2)
        )
        ** 0.5
    )
    trial_func = fenics.TrialFunction(func_space)
    test_func = fenics.TestFunction(func_space)
    C_form = trial_func * test_func * ufl.dx
    G_form = ufl.inner(ufl.grad(test_func), ufl.grad(trial_func)) * ufl.dx
    K_form = C_form + (1 / kappa ** 2) * G_form
    K_matrix = fenics.assemble(K_form)
    if alpha > 1:
        C_matrix = fenics.assemble(C_form)
    if alpha > 2:
        C_op = fenics_matrix_to_linear_operator(C_matrix)
    if alpha % 2 == 0:
        inv_K_op = cholmod_factor_to_inv_linear_operator(
            *psd_fenics_matrix_to_cholmod_factor_and_shape(K_matrix)
        )
        sqrt_C_op = cholmod_factor_to_sqrt_linear_operator(
            *psd_fenics_matrix_to_cholmod_factor_and_shape(C_matrix)
        )
        inner_op = inv_K_op @ sqrt_C_op
    elif alpha == 1:
        inner_op = cholmod_factor_to_sqrt_inv_linear_operator(
            *psd_fenics_matrix_to_cholmod_factor_and_shape(K_matrix)
        )
    else:
        K_cholmod_factor, shape = psd_fenics_matrix_to_cholmod_factor_and_shape(
            K_matrix
        )
        inv_K_op = cholmod_factor_to_inv_linear_operator(K_cholmod_factor, shape)
        inner_op = cholmod_factor_to_sqrt_inv_linear_operator(K_cholmod_factor, shape)

    if alpha <= 2:
        return inner_op * eta
    else:
        outer_op = (inv_K_op @ C_op) ** (alpha - 2)
        return outer_op @ inner_op * eta


def construct_forward_func_and_derivatives(
    z_func_space,
    x_func_space,
    define_forms,
    define_boundary_conditions,
    observation_coordinates,
    prior_covar_sqrt=None,
    bilinear_form_is_symmetric=False,
):

    # Get problem dimensions
    dim_x = x_func_space.dim()
    dim_y = observation_coordinates.shape[0]
    dim_z = z_func_space.dim()

    # Set up observation operator
    # (point observations at DOFs closest to observation coordinates)
    dof_coordinates = x_func_space.tabulate_dof_coordinates()
    observation_dof_indices = np.argmin(
        ((dof_coordinates[None, :, :] - observation_coordinates[:, None]) ** 2).sum(-1),
        -1,
    )
    observation_matrix = np.zeros((dim_y, dim_x))
    observation_matrix[np.arange(dim_y), observation_dof_indices] = 1

    # Default to identity prior covariance if prior_covar_sqrt is None
    if prior_covar_sqrt is None:
        prior_covar_sqrt = LinearOperator(
            shape=(dim_z, dim_z), matvec=lambda x: x, rmatvec=lambda x: x
        )

    # Construct function objects
    z = fenics.Function(z_func_space)
    x = fenics.Function(x_func_space)
    x_trial = fenics.TrialFunction(x_func_space)
    x_test = fenics.TestFunction(x_func_space)
    h = fenics.Function(x_func_space)
    v_H = fenics.Function(x_func_space)
    k = fenics.Function(x_func_space)
    dAx_dz_m = fenics.Function(x_func_space)
    m_list = [fenics.Function(z_func_space) for _ in range(dim_y)]
    h_A_inv_list = [fenics.Function(x_func_space) for _ in range(dim_y)]
    A_inv_dAx_dz_m_list = [fenics.Function(x_func_space) for _ in range(dim_y)]

    # Construct bilinear and linear forms defining variational form of problem
    A, b = define_forms(z, x_test, x_trial)
    boundary_conditions = define_boundary_conditions(x_func_space)

    # Precompute additional forms required for calculating first-derivatives
    if not bilinear_form_is_symmetric:
        adjoint_A = fenics.adjoint(A)
    dAx_dz = fenics.derivative(A(x_test, x, coefficients={}), z)
    adjoint_dAx_dz = fenics.adjoint(dAx_dz)
    k_dAx_dz = fenics.derivative(A(k, x, coefficients={}), z)

    # Create homogenized boundary conditions for solving in adjoint pass
    homogenized_boundary_conditions = define_boundary_conditions()
    for boundary_condition in homogenized_boundary_conditions:
        boundary_condition.homogenize()

    # Preallocate numpy array for storing Jacobian
    dy_dv = np.full((dim_y, dim_z), np.nan)

    # Precompute forms required for calculating second-derivatives
    g_1, g_2, g_3 = 0, 0, 0
    for h_A_inv, m, A_inv_dAx_dz_m in zip(h_A_inv_list, m_list, A_inv_dAx_dz_m_list):
        g_1 += fenics.derivative(A(h_A_inv, A_inv_dAx_dz_m, coefficients={}), z)
        g_2 += fenics.derivative(A(h_A_inv, x_trial, coefficients={}), z, m)
        g_3 -= fenics.derivative(
            fenics.derivative(A(h_A_inv, x, coefficients={}), z, m), z
        )

    def solution_func(v_array):
        z_array = prior_covar_sqrt @ v_array
        z.vector().set_local(z_array)
        fenics.solve(A == b, x, bcs=boundary_conditions)
        return x.vector().get_local()

    def forward_func(v_array):
        return solution_func(v_array)[observation_dof_indices]

    def _get_solvers_and_y():
        if not bilinear_form_is_symmetric:
            # Homogenized and original boundary conditions have equivalent effect on matrix
            # corresponding to assembled bilinear form
            A_solver = construct_solver(A, boundary_conditions)
            b_vector = fenics.assemble(b)
            adjoint_A_solver = construct_solver(
                adjoint_A, homogenized_boundary_conditions
            )
            solve_with_boundary_conditions(A_solver, boundary_conditions, b_vector, x)
        else:
            A_matrix, b_vector = fenics.assemble_system(A, b, boundary_conditions)
            A_solver = fenics.LUSolver(A_matrix)
            A_solver.parameters["symmetric"] = True
            adjoint_A_solver = A_solver  # A is symmetric therefore adjoint(A) == A
            A_solver.solve(x.vector(), b_vector)
        y = (x.vector()).get_local()[observation_dof_indices]
        return A_solver, adjoint_A_solver, y

    def vjp_forward_func(v_array):
        z_array = prior_covar_sqrt @ v_array
        z.vector().set_local(z_array)
        _, adjoint_A_solver, y = _get_solvers_and_y()

        def vjp(v):
            v_H.vector().set_local(v @ observation_matrix)
            solve_with_boundary_conditions(
                adjoint_A_solver, homogenized_boundary_conditions, v_H.vector(), k
            )
            return -prior_covar_sqrt.T @ fenics.assemble(k_dAx_dz).get_local()

        return vjp, y

    def _jacob_forward_func(v_array):
        z_array = prior_covar_sqrt @ v_array
        z.vector().set_local(z_array)
        A_solver, adjoint_A_solver, y = _get_solvers_and_y()
        adjoint_dAx_dz_matrix = fenics.assemble(adjoint_dAx_dz)
        for dy_dv_row, h_arr, h_A_inv in zip(dy_dv, observation_matrix, h_A_inv_list):
            h.vector().set_local(h_arr)
            solve_with_boundary_conditions(
                adjoint_A_solver, homogenized_boundary_conditions, h.vector(), h_A_inv
            )
            dy_dv_row[:] = (
                -prior_covar_sqrt.T
                @ (adjoint_dAx_dz_matrix * h_A_inv.vector()).get_local()
            )
        return dy_dv, y, A_solver, adjoint_A_solver, adjoint_dAx_dz_matrix

    def jacob_forward_func(v_array):
        return _jacob_forward_func(v_array)[:2]

    def mhp_forward_func(v_array):
        (
            dy_dv,
            y,
            A_solver,
            adjoint_A_solver,
            adjoint_dAx_dz_matrix,
        ) = _jacob_forward_func(v_array)

        def mhp(matrix):
            for m_arr, m, A_inv_dAx_dz_m in zip(matrix, m_list, A_inv_dAx_dz_m_list):
                m.vector().set_local(prior_covar_sqrt @ m_arr)
                adjoint_dAx_dz_matrix.transpmult(m.vector(), dAx_dz_m.vector())
                solve_with_boundary_conditions(
                    A_solver,
                    homogenized_boundary_conditions,
                    dAx_dz_m.vector(),
                    A_inv_dAx_dz_m,
                )
            g = (
                fenics.assemble(g_1 + g_3)
                + adjoint_dAx_dz_matrix
                * solve_with_boundary_conditions(
                    adjoint_A_solver,
                    homogenized_boundary_conditions,
                    fenics.assemble(g_2),
                    x_func_space,
                ).vector()
            )
            return prior_covar_sqrt.T @ g.get_local()

        return mhp, dy_dv, y

    return (
        solution_func,
        forward_func,
        vjp_forward_func,
        jacob_forward_func,
        mhp_forward_func,
        (dim_x, dim_y, dim_z),
    )


def construct_constrained_system(
    y_obs,
    forward_func,
    jacob_forward_func,
    mhp_forward_func,
    dim_y,
    dim_z,
    generate_σ,
    grad_generate_σ,
    hessian_generate_σ,
):
    def constr(q):
        u, v, n = q[0], q[1 : dim_z + 1], q[dim_z + 1 :]
        return forward_func(v) + generate_σ(u) * n - y_obs

    def jacob_constr(q):
        u, v, n = q[0], q[1 : dim_z + 1], q[dim_z + 1 :]
        dσ_du, σ = grad_generate_σ(u)
        dy_dv, y_mean = jacob_forward_func(v)
        c = y_mean + σ * n - y_obs
        dc_dq = np.concatenate([dσ_du * n[:, None], dy_dv, σ * np.identity(dim_y)], 1)
        return dc_dq, c

    def mhp_constr(q):
        u, v, n = q[0], q[1 : dim_z + 1], q[dim_z + 1 :]
        d2σ_du2, dσ_du, σ = hessian_generate_σ(u)
        mhp_y_v, dy_dv, y_mean = mhp_forward_func(v)
        c = y_mean + σ * n - y_obs
        dc_dq = np.concatenate([dσ_du * n[:, None], dy_dv, σ * np.identity(dim_y)], 1)

        def mhp(m):
            return np.concatenate(
                [
                    np.array(
                        [d2σ_du2 * n @ m[:, 0] + dσ_du * m[:, dim_z + 1 :].trace()]
                    ),
                    mhp_y_v(m[:, 1 : dim_z + 1]),
                    dσ_du * m[:, 0],
                ]
            )

        return mhp, dc_dq, c

    def neg_log_dens(q):
        return (q ** 2).sum() / 2

    def grad_neg_log_dens(q):
        return q, (q ** 2).sum() / 2

    return mici.systems.DenseConstrainedEuclideanMetricSystem(
        constr=constr,
        jacob_constr=jacob_constr,
        mhp_constr=mhp_constr,
        neg_log_dens=neg_log_dens,
        grad_neg_log_dens=grad_neg_log_dens,
        dens_wrt_hausdorff=False,
    )


def construct_euclidean_system(
    y_obs, forward_func, vjp_forward_func, dim_y, dim_z, generate_σ, grad_generate_σ
):
    def neg_log_dens(q):
        u, v = q[0], q[1 : dim_z + 1]
        σ = generate_σ(u)
        y_mean = forward_func(v)
        return (((y_mean - y_obs) / σ) ** 2 / 2 + np.log(σ)).sum() + (q ** 2).sum() / 2

    def grad_neg_log_dens(q):
        u, v = q[0], q[1 : dim_z + 1]
        dσ_du, σ = grad_generate_σ(u)
        vjp, y_mean = vjp_forward_func(v)
        sum_residual_over_σ_sq = (((y_mean - y_obs) / σ) ** 2).sum()
        grad = np.concatenate(
            [
                np.array([dσ_du / σ * (dim_y - sum_residual_over_σ_sq) + u]),
                vjp(y_mean - y_obs) / σ ** 2 + v,
            ]
        )
        return (
            grad,
            sum_residual_over_σ_sq / 2 + dim_y * np.log(σ) + (q ** 2).sum() / 2,
        )

    return mici.systems.EuclideanMetricSystem(
        neg_log_dens=neg_log_dens, grad_neg_log_dens=grad_neg_log_dens
    )

