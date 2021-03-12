"""Inference of coefficient field in Poisson equation (elliptic PDE)."""

import os
import warnings
import logging
import numpy as np
import fenics
import ufl
import mlift.pde as pde
import mlift.example_models.utils as utils
import mici


mesh, observation_coordinates = pde.construct_mesh_and_observation_coordinates(
    "circle", 12, 3
)
x_func_space = fenics.FunctionSpace(mesh, "Lagrange", 1)
z_func_space = fenics.FunctionSpace(mesh, "Lagrange", 1)


def define_forms(z, x_test, x):
    f = fenics.Constant(100.0)
    return (
        ufl.inner(ufl.exp(z) * ufl.grad(x), ufl.grad(x_test)) * ufl.dx,
        f * x_test * ufl.dx,
    )


def define_boundary_conditions(x_func_space):
    def boundary(s, on_boundary):
        return on_boundary

    return [fenics.DirichletBC(x_func_space, 0.0, boundary)]


prior_covar_sqrt = pde.construct_matern_covar_sqrt_operator(
    z_func_space, length_scale=1.0, amplitude=1.0, order=2
)

(
    solution_func,
    forward_func,
    vjp_forward_func,
    jacob_forward_func,
    mhp_forward_func,
    (dim_x, dim_y, dim_z),
) = pde.construct_forward_func_and_derivatives(
    z_func_space=z_func_space,
    x_func_space=x_func_space,
    define_forms=define_forms,
    define_boundary_conditions=define_boundary_conditions,
    observation_coordinates=observation_coordinates,
    prior_covar_sqrt=prior_covar_sqrt,
    bilinear_form_is_symmetric=True,
)


log_σ_prior_std = 3.
log_σ_prior_mean = 0.

def generate_σ(u):
    return np.exp(log_σ_prior_std * u + log_σ_prior_mean)

def grad_generate_σ(u):
    σ = generate_σ(u)
    return log_σ_prior_std * σ, σ

def hessian_generate_σ(u):
    σ = generate_σ(u)
    return log_σ_prior_std**2 * σ, log_σ_prior_std * σ, σ


def sample_initial_states(rng, data, num_chain=4, algorithm="chmc"):
    """Sample initial states from prior."""
    init_states = []
    for _ in range(num_chain):
        u = rng.standard_normal()
        v = rng.standard_normal(dim_z)
        if algorithm == "chmc":
            y_mean = forward_func(v)
            σ = generate_σ(u)
            n = (data["y_obs"] - y_mean) / σ
            q = np.concatenate((np.array([u]), v, n))
        else:
            q = np.concatenate((np.array([u]), v))
        init_states.append(q)
    return init_states


if __name__ == "__main__":

    # Process command line arguments defining experiment parameters

    parser = utils.set_up_argparser_with_standard_arguments(
        "Run Poisson PDE model simulated data experiment"
    )
    parser.add_argument(
        "--obs-noise-std",
        type=float,
        default=1e-2,
        help="Standard deviation of observation noise to use in simulated data",
    )
    args = parser.parse_args()

    # Load data

    data = dict(np.load(os.path.join(args.data_dir, "poisson-simulated-data.npz")))
    data["y_obs"] = data["y_mean"] + args.obs_noise_std * data["n_obs"]

    # Set up seeded random number generator

    rng = np.random.default_rng(args.seed)

    # Define variables to be traced

    def trace_func(state):
        u, v = state.pos[0], state.pos[1 : 1 + dim_z]
        σ = generate_σ(u)
        z = prior_covar_sqrt @ v
        x = solution_func(v)
        return {"σ": σ, "z_mean": z.mean(), "z_std": z.std(), "z": z, "x": x}


    # Disable fenics logging to prevent interference with progress meter display

    fenics.set_log_active(False)
    logging.getLogger("UFL").setLevel(logging.WARNING)
    logging.getLogger("FFC").setLevel(logging.WARNING)

    # Disable runtime warnings to prevent interference with progress meter display

    warnings.filterwarnings("ignore", category=RuntimeWarning) 

    # Run experiment

    constrained_system_kwargs = pde.construct_constrained_system_kwargs(
        y_obs=data["y_obs"],
        forward_func=forward_func,
        jacob_forward_func=jacob_forward_func,
        mhp_forward_func=mhp_forward_func,
        dim_y=dim_y,
        dim_z=dim_z,
        generate_σ=generate_σ,
        grad_generate_σ=grad_generate_σ,
        hessian_generate_σ=hessian_generate_σ,
    )

    euclidean_system_kwargs = pde.construct_euclidean_system_kwargs(
        y_obs=data["y_obs"],
        forward_func=forward_func,
        vjp_forward_func=vjp_forward_func,
        dim_y=dim_y,
        dim_z=dim_z,
        generate_σ=generate_σ,
        grad_generate_σ=grad_generate_σ,
    )

    final_states, traces, stats, summary_dict, sampler = utils.run_experiment(
        args=args,
        data=data,
        rng=rng,
        experiment_name="poisson",
        dir_prefix=f"σ_{args.obs_noise_std:.0e}",
        var_names=["σ", "z_mean", "z_std"],
        var_trace_func=trace_func,
        constrained_system_class=mici.systems.DenseConstrainedEuclideanMetricSystem,
        constrained_system_kwargs=constrained_system_kwargs,
        euclidean_system_class=mici.systems.EuclideanMetricSystem,
        euclidean_system_kwargs=euclidean_system_kwargs,
        sample_initial_states=sample_initial_states,
        precompile_jax_functions=False
    )
