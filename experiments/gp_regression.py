"""Gaussian process regression with Gaussian observation noise"""

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
    return squared_exponential_covariance(data["x"], params) + params[
        "σ"
    ] ** 2 * np.identity(dim_y)


def prior_neg_log_dens(u):
    return (
        np.exp(2 * u[0]) / 2
        - u[0]
        + (
            λ_prior_shape * polygamma(1, λ_prior_shape) ** 0.5 * u[1:-1]
            + λ_prior_scale / np.exp(polygamma(1, λ_prior_shape) ** 0.5 * u[1:-1])
        ).sum()
        + np.exp(2 * u[-1]) / 2
        - u[-1]
    )


def extended_prior_neg_log_dens(q, data):
    dim_u = data["dim_u"]
    u, n = q[:dim_u], q[dim_u:]
    return (n ** 2).sum() / 2 + prior_neg_log_dens(u)


def posterior_neg_log_dens(u, data):
    covar = covar_func(u, data)
    chol_covar = np.linalg.cholesky(covar)
    return (
        data["y_obs"] @ sla.cho_solve((chol_covar, True), data["y_obs"]) / 2
        + np.log(chol_covar.diagonal()).sum()
        + prior_neg_log_dens(u)
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
        if args.algorithm == "chmc":
            chol_covar = onp.linalg.cholesky(covar_func(u, data))
            n = sla.solve_triangular(chol_covar, data["y_obs"], lower=True)
            q = onp.concatenate((u, onp.asarray(n)))
            assert (
                abs(chol_covar @ n - data["y_obs"]).max()
                < args.projection_solver_warm_up_constraint_tol
            )
        else:
            q = u
        init_states.append(q)
    return init_states


if __name__ == "__main__":

    # Process command line arguments defining experiment parameters

    parser = common.set_up_argparser_with_standard_arguments(
        "Run Gaussian process regression (Gaussian likelihood) experiment"
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
    args = parser.parse_args()

    # Set up seeded random number generator

    rng = onp.random.default_rng(args.seed)

    # Load data

    data = common.load_regression_data(args, rng)
    dim_u = data["x"].shape[1] + 2
    data["dim_u"] = dim_u
    dim_y = data["y_obs"].shape[0]

    # Define variables to be traced

    def trace_func(state):
        u = state.pos[:dim_u]
        params = generate_params(u)
        return {**params, "u": u}

    # Run experiment

    (
        neg_log_dens_chmc,
        grad_neg_log_dens_chmc,
    ) = mlift.construct_mici_system_neg_log_dens_functions(
        lambda q: extended_prior_neg_log_dens(q, data)
    )

    final_states, traces, stats, summary_dict = common.run_experiment(
        args=args,
        data=data,
        dim_u=dim_u,
        rng=rng,
        experiment_name="gp_regression",
        dir_prefix=f"{args.dataset}_data",
        var_names=["α", "λ", "σ"],
        var_trace_func=trace_func,
        posterior_neg_log_dens=posterior_neg_log_dens,
        constrained_system_class=mlift.GaussianProcessModelSystem,
        constrained_system_kwargs={
            "covar_func": covar_func,
            "data": data,
            "dim_u": dim_u,
            "neg_log_dens": neg_log_dens_chmc,
            "grad_neg_log_dens": grad_neg_log_dens_chmc,
        },
        sample_initial_states=sample_initial_states,
    )

