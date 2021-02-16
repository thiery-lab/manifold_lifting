"""Gaussian process regression with Gaussian observation noise"""

import os
import pickle
import numpy as onp
import jax.config
import jax.numpy as np
import jax.lax as lax
import jax.api as api
import jax.scipy.linalg as sla
import mlift
from mlift.distributions import half_normal, inverse_gamma
from experiments import common

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


prior_specifications = {
    "α": common.PriorSpecification(distribution=half_normal(3)),
    "λ": common.PriorSpecification(
        shape=lambda data: data["x"].shape[1], distribution=inverse_gamma(4, 10)
    ),
    "σ": common.PriorSpecification(distribution=half_normal(1)),
}


(
    compute_dim_u,
    generate_params,
    prior_neg_log_dens,
    sample_from_prior,
) = common.set_up_prior(prior_specifications)


def squared_exponential_covariance(x, params):
    def sq_exp(x1, x2):
        return np.exp(-(((x1 - x2) / params["λ"]) ** 2).sum() / 2)

    return params["α"] * api.vmap(lambda x1: api.vmap(lambda x2: sq_exp(x1, x2))(x))(x)


def covar_func(u, data):
    dim_y = data["y_obs"].shape[0]
    params = generate_params(u, data)
    return squared_exponential_covariance(data["x"], params) + params[
        "σ"
    ] ** 2 * np.identity(dim_y)


def extended_prior_neg_log_dens(q, data):
    dim_u = compute_dim_u(data)
    u, n = q[:dim_u], q[dim_u:]
    return prior_neg_log_dens(u, data) + (n ** 2).sum() / 2


def posterior_neg_log_dens(u, data):
    covar = covar_func(u, data)
    chol_covar = np.linalg.cholesky(covar)
    return prior_neg_log_dens(u, data) + (
        data["y_obs"] @ sla.cho_solve((chol_covar, True), data["y_obs"]) / 2
        + np.log(chol_covar.diagonal()).sum()
    )


def sample_initial_states(rng, args, data):
    """Sample initial states from prior."""
    init_states = []
    for _ in range(args.num_chain):
        u = sample_from_prior(rng, data)
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
    dim_u = compute_dim_u(data)
    dim_y = data["y_obs"].shape[0]

    # Define variables to be traced

    def trace_func(state):
        u = state.pos[:dim_u]
        params = generate_params(u, data)
        return {**params, "u": u}

    # Run experiment

    final_states, traces, stats, summary_dict, sampler = common.run_experiment(
        args=args,
        data=data,
        dim_u=dim_u,
        rng=rng,
        experiment_name="gp_regression",
        dir_prefix=f"{args.dataset}_data_subsampled_by_{args.data_subsample}",
        var_names=list(prior_specifications.keys()),
        var_trace_func=trace_func,
        posterior_neg_log_dens=posterior_neg_log_dens,
        extended_prior_neg_log_dens=extended_prior_neg_log_dens,
        constrained_system_class=mlift.GaussianProcessModelSystem,
        constrained_system_kwargs={
            "covar_func": covar_func,
            "data": data,
            "dim_u": dim_u,
        },
        sample_initial_states=sample_initial_states,
    )

