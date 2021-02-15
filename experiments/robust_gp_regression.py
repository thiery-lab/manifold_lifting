"""Gaussian process regression with Student's T observation noise (ν = 4)"""

import os
import pickle
import numpy as onp
import jax.config
import jax.numpy as np
import jax.lax as lax
import jax.api as api
import jax.scipy.linalg as sla
from jax.scipy.special import polygamma
import mlift
from mlift.transforms import normal_to_students_t_4, students_t_4_to_normal
from experiments import common

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


λ_prior_shape, λ_prior_scale = (4, 10)


def generate_params(u):
    return {
        "α": np.exp(u[0]) * 3,
        "λ": np.exp(u[1:-1] * polygamma(1, λ_prior_shape) ** 0.5),
        "σ": np.exp(u[-1]),
    }


def squared_exponential_covariance(x, params):
    def sq_exp(x1, x2):
        return np.exp(-(((x1 - x2) / params["λ"]) ** 2).sum() / 2)

    return params["α"] * api.vmap(lambda x1: api.vmap(lambda x2: sq_exp(x1, x2))(x))(x)


def covar_func(u, data):
    dim_y = data["y_obs"].shape[0]
    params = generate_params(u)
    return squared_exponential_covariance(data["x"], params) + data[
        "covar_jitter"
    ] * np.identity(dim_y)


def noise_scale_func(u):
    return generate_params(u)["σ"]


noise_transform_func = normal_to_students_t_4
inverse_noise_transform_func = students_t_4_to_normal


def prior_neg_log_dens(u, v):
    return (
        np.exp(2 * u[0]) / 2
        - u[0]
        + (
            λ_prior_shape * polygamma(1, λ_prior_shape) ** 0.5 * u[1:-1]
            + λ_prior_scale / np.exp(polygamma(1, λ_prior_shape) ** 0.5 * u[1:-1])
        ).sum()
        + np.exp(2 * u[-1]) / 2
        - u[-1]
        + (v ** 2).sum() / 2
    )


def extended_prior_neg_log_dens(q, data):
    dim_u = data["dim_u"]
    dim_y = data["y_obs"].shape[0]
    u, v, n = q[:dim_u], q[dim_u : dim_u + dim_y], q[dim_u + dim_y :]
    return (n ** 2).sum() / 2 + prior_neg_log_dens(u, v)


def posterior_neg_log_dens(q, data):
    dim_u = data["dim_u"]
    u, v = q[:dim_u], q[dim_u:]
    covar = covar_func(u, data)
    chol_covar = np.linalg.cholesky(covar)
    σ = noise_scale_func(u)
    t = (chol_covar @ v - data["y_obs"]) / σ
    return ((5 / 2) * np.log(1 + t ** 2 / 4) + np.log(σ)).sum() + prior_neg_log_dens(
        u, v
    )


def sample_initial_states(rng, args, data):
    """Sample initial states from prior."""
    init_states = []
    for _ in range(args.num_chain):
        u = onp.concatenate(
            (
                onp.log(abs(rng.standard_normal()))[None],
                -onp.log(
                    rng.gamma(
                        shape=λ_prior_shape,
                        scale=1 / λ_prior_scale,
                        size=data["x"].shape[1],
                    )
                ),
                onp.log(abs(rng.standard_normal()))[None],
            )
        )
        v = rng.standard_normal(dim_y)
        if args.algorithm == "chmc":
            y_mean = onp.linalg.cholesky(covar_func(u, data)) @ v
            n = inverse_noise_transform_func(
                (data["y_obs"] - y_mean) / noise_scale_func(u)
            )
            q = onp.concatenate((u, v, onp.asarray(n)))
            assert (
                abs(
                    y_mean
                    + noise_scale_func(u) * noise_transform_func(n)
                    - data["y_obs"]
                ).max()
                < args.projection_solver_warm_up_constraint_tol
            )
        else:
            q = onp.concatenate((u, v))
        init_states.append(q)
    return init_states


if __name__ == "__main__":

    # Process command line arguments defining experiment parameters

    parser = common.set_up_argparser_with_standard_arguments(
        "Run Gaussian process regression (Student's T likelihood, ν = 4) experiment"
    )
    parser.add_argument(
        "--dataset",
        default="yacht",
        choices=("yacht", "slump", "marthe"),
        help="Which data set to perform inference with",
    )
    parser.add_argument(
        "--data-subsample",
        type=int,
        default=1,
        help="Factor to subsample data by (1 corresponds to no subsampling)",
    )
    parser.add_argument(
        "--covar-jitter",
        type=float,
        default=1e-8,
        help="Scale of 'jitter' term added to covariance diagonal for numerical stability",
    )
    args = parser.parse_args()

    # Set up seeded random number generator

    rng = onp.random.default_rng(args.seed)

    # Load data

    data = common.load_regression_data(args, rng)
    data["covar_jitter"] = args.covar_jitter
    dim_u = data["x"].shape[1] + 2
    data["dim_u"] = dim_u
    dim_y = data["y_obs"].shape[0]

    # Define variables to be traced

    def trace_func(state):
        u, v = state.pos[:dim_u], state.pos[dim_u : dim_u + dim_y]
        params = generate_params(u)
        return {**params, "u": u, "v": v}

    # Run experiment

    (
        neg_log_dens_chmc,
        grad_neg_log_dens_chmc,
    ) = mlift.construct_mici_system_neg_log_dens_functions(
        lambda q: extended_prior_neg_log_dens(q, data)
    )

    final_states, traces, stats, summary_dict, sampler = common.run_experiment(
        args=args,
        data=data,
        dim_u=dim_u,
        rng=rng,
        experiment_name="robust_gp_regression",
        dir_prefix=f"{args.dataset}_data",
        var_names=["α", "λ", "σ"],
        var_trace_func=trace_func,
        posterior_neg_log_dens=posterior_neg_log_dens,
        constrained_system_class=mlift.GeneralGaussianProcessModelSystem,
        constrained_system_kwargs={
            "covar_func": covar_func,
            "noise_scale_func": noise_scale_func,
            "noise_transform_func": noise_transform_func,
            "data": data,
            "dim_u": dim_u,
            "neg_log_dens": neg_log_dens_chmc,
            "grad_neg_log_dens": grad_neg_log_dens_chmc,
        },
        sample_initial_states=sample_initial_states,
    )

