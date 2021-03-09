"""Gaussian process regression with Student's T observation noise (ν = 4)"""

import os
import pickle
import numpy as onp
import jax.config
import jax.numpy as np
import jax.lax as lax
import jax.api as api
from mlift.systems import GeneralGaussianProcessModelSystem
from mlift.transforms import standard_normal_to_students_t
from mlift.distributions import half_normal, inverse_gamma, students_t
from mlift.prior import PriorSpecification, set_up_prior
import mlift.example_models.utils as utils

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


prior_specifications = {
    "α": PriorSpecification(distribution=half_normal(3)),
    "λ": PriorSpecification(
        shape=lambda data: data["x"].shape[1], distribution=inverse_gamma(4, 10)
    ),
    "σ": PriorSpecification(distribution=half_normal(1)),
}


compute_dim_u, generate_params, prior_neg_log_dens, sample_from_prior = set_up_prior(
    prior_specifications
)


def squared_exp_covar(x, params):
    def sq_exp(x1, x2):
        return np.exp(-(((x1 - x2) / params["λ"]) ** 2).sum() / 2)

    return params["α"] * api.vmap(lambda x1: api.vmap(lambda x2: sq_exp(x1, x2))(x))(x)


def covar_func(u, data):
    dim_y = data["y_obs"].shape[0]
    params = generate_params(u, data)
    return squared_exp_covar(data["x"], params) + data["covar_jitter"] * np.identity(
        dim_y
    )


def noise_scale_func(u, data):
    return generate_params(u, data)["σ"]


noise_transform_func = standard_normal_to_students_t(0, 1, 4).forward
inverse_noise_transform_func = standard_normal_to_students_t(0, 1, 4).backward


def extended_prior_neg_log_dens(q, data):
    dim_u = compute_dim_u(data)
    dim_y = data["y_obs"].shape[0]
    u, v, n = q[:dim_u], q[dim_u : dim_u + dim_y], q[dim_u + dim_y :]
    return prior_neg_log_dens(u, data) + (v ** 2).sum() / 2 + (n ** 2).sum() / 2


def posterior_neg_log_dens(q, data):
    dim_u = compute_dim_u(data)
    u, v = q[:dim_u], q[dim_u:]
    covar = covar_func(u, data)
    chol_covar = np.linalg.cholesky(covar)
    σ = noise_scale_func(u, data)
    return (
        prior_neg_log_dens(u, data)
        + (v ** 2).sum() / 2
        + students_t(chol_covar @ v, σ, 4).neg_log_dens(data["y_obs"], True)
    )


def sample_initial_states(rng, data, num_chain=4, algorithm="chmc"):
    """Sample initial states from prior."""
    init_states = []
    dim_y = data["y_obs"].shape[0]
    for _ in range(num_chain):
        u = sample_from_prior(rng, data)
        v = rng.standard_normal(dim_y)
        if algorithm == "chmc":
            y_mean = onp.linalg.cholesky(covar_func(u, data)) @ v
            n = inverse_noise_transform_func(
                (data["y_obs"] - y_mean) / noise_scale_func(u, data)
            )
            q = onp.concatenate((u, v, onp.asarray(n)))
        else:
            q = onp.concatenate((u, v))
        init_states.append(q)
    return init_states


if __name__ == "__main__":

    # Process command line arguments defining experiment parameters

    parser = utils.set_up_argparser_with_standard_arguments(
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

    data = utils.load_regression_data(args, rng)
    data["covar_jitter"] = args.covar_jitter
    dim_u = compute_dim_u(data)
    dim_y = data["y_obs"].shape[0]

    # Define variables to be traced

    trace_func = utils.construct_trace_func(generate_params, data, dim_u, dim_v=dim_y)

    # Run experiment

    final_states, traces, stats, summary_dict, sampler = utils.run_experiment(
        args=args,
        data=data,
        rng=rng,
        experiment_name="robust_gp_regression",
        dir_prefix=f"{args.dataset}_data_subsampled_by_{args.data_subsample}",
        var_names=list(prior_specifications.keys()),
        var_trace_func=trace_func,
        posterior_neg_log_dens=posterior_neg_log_dens,
        extended_prior_neg_log_dens=extended_prior_neg_log_dens,
        constrained_system_class=GeneralGaussianProcessModelSystem,
        constrained_system_kwargs={
            "covar_func": covar_func,
            "noise_scale_func": noise_scale_func,
            "noise_transform_func": noise_transform_func,
            "data": data,
            "dim_u": dim_u,
        },
        sample_initial_states=sample_initial_states,
    )

