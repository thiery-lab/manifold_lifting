"""Autoregressive order-K (AR-K) benchmark model

Model definition and data taken from:

https://github.com/stan-dev/stat_comp_benchmarks/tree/master/benchmarks/arK
"""

import os
import numpy as onp
import jax.config
import jax.numpy as np
import jax.lax as lax
import jax.api as api
import mlift
from mlift.distributions import normal, uniform, half_cauchy
from experiments import common

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


prior_specifications = {
    "α": common.PriorSpecification(distribution=normal(0, 10)),
    "β": common.PriorSpecification(
        shape=lambda data: data["max_lag"], distribution=normal(0, 10)
    ),
    "σ": common.PriorSpecification(distribution=half_cauchy(2.5)),
}

(
    compute_dim_u,
    generate_params,
    prior_neg_log_dens,
    sample_from_prior,
) = common.set_up_prior(prior_specifications)


def generate_from_model(u, data):
    params = generate_params(u, data)
    x = params["α"] + (data["y_windows"] * params["β"]).sum(-1)
    return params, x


def generate_y(u, n, data):
    params, x = generate_from_model(u, data)
    y = x + params["σ"] * n
    return y


def extended_prior_neg_log_dens(q, data):
    dim_u = compute_dim_u(data)
    u, n = q[:dim_u], q[dim_u:]
    return prior_neg_log_dens(u, data) + (n ** 2).sum() / 2


def posterior_neg_log_dens(u, data):
    params, x = generate_from_model(u, data)
    return prior_neg_log_dens(u, data) + (
        ((data["y_obs"] - x) / params["σ"]) ** 2 / 2 + np.log(params["σ"])
    ).sum()


def sample_initial_states(rng, args, data):
    """Sample initial states from prior."""
    init_states = []
    for _ in range(args.num_chain):
        u = sample_from_prior(rng, data)
        if args.algorithm == "chmc":
            params, x = generate_from_model(u, data)
            n = (data["y_obs"] - x) / params["σ"]
            q = onp.concatenate((u, onp.asarray(n)))
            assert (
                abs(x + params["σ"] * n - data["y_obs"]).max()
                < args.projection_solver_warm_up_constraint_tol
            )
        else:
            q = u
        init_states.append(q)
    return init_states


if __name__ == "__main__":

    # Process command line arguments defining experiment parameters

    parser = common.set_up_argparser_with_standard_arguments(
        "Run autoregressive order-K (AR-K) benchmark model experiment "
    )
    args = parser.parse_args()

    # Load data

    data = dict(np.load(os.path.join(args.data_dir, "ar-k-benchmark-data.npz")))
    dim_u = compute_dim_u(data)

    # Set up seeded random number generator

    rng = onp.random.default_rng(args.seed)

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
        experiment_name="garch",
        var_names=list(prior_specifications.keys()),
        var_trace_func=trace_func,
        posterior_neg_log_dens=posterior_neg_log_dens,
        extended_prior_neg_log_dens=extended_prior_neg_log_dens,
        constrained_system_class=mlift.IndependentAdditiveNoiseModelSystem,
        constrained_system_kwargs={
            "generate_y": generate_y,
            "data": data,
            "dim_u": dim_u,
        },
        sample_initial_states=sample_initial_states,
    )

