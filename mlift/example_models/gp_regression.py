"""Gaussian process regression with Gaussian observation noise"""

import numpy as onp
import jax
import jax.config
import jax.numpy as np
import jax.scipy.linalg as sla
from mlift.systems import GaussianProcessModelSystem
from mlift.distributions import half_normal, inverse_gamma
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

    return params["α"] * jax.vmap(lambda x1: jax.vmap(lambda x2: sq_exp(x1, x2))(x))(x)


def covar_func(u, data):
    dim_y = data["y_obs"].shape[0]
    params = generate_params(u, data)
    return squared_exp_covar(data["x"], params) + params["σ"] ** 2 * np.identity(dim_y)


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


def sample_initial_states(rng, data, num_chain=4, algorithm="chmc"):
    """Sample initial states from prior."""
    init_states = []
    for _ in range(num_chain):
        u = sample_from_prior(rng, data)
        if algorithm == "chmc":
            chol_covar = onp.linalg.cholesky(covar_func(u, data))
            n = sla.solve_triangular(chol_covar, data["y_obs"], lower=True)
            q = onp.concatenate((u, onp.asarray(n)))
        else:
            q = onp.asarray(u)
        init_states.append(q)
    return init_states


if __name__ == "__main__":

    # Process command line arguments defining experiment parameters

    parser = utils.set_up_argparser_with_standard_arguments(
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

    data = utils.load_regression_data(args, rng)
    dim_u = compute_dim_u(data)
    dim_y = data["y_obs"].shape[0]

    # Define variables to be traced

    trace_func = utils.construct_trace_func(generate_params, data, dim_u)

    # Run experiment

    final_states, traces, stats, summary_dict, sampler = utils.run_experiment(
        args=args,
        data=data,
        rng=rng,
        experiment_name="gp_regression",
        dir_prefix=f"{args.dataset}_data_subsampled_by_{args.data_subsample}",
        var_names=list(prior_specifications.keys()),
        var_trace_func=trace_func,
        posterior_neg_log_dens=posterior_neg_log_dens,
        extended_prior_neg_log_dens=extended_prior_neg_log_dens,
        constrained_system_class=GaussianProcessModelSystem,
        constrained_system_kwargs={
            "covar_func": covar_func,
            "data": data,
            "dim_u": dim_u,
        },
        sample_initial_states=sample_initial_states,
    )

