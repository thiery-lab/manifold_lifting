"""Autoregressive moving average (ARMA) benchmark model

Model definition and data taken from:

https://github.com/stan-dev/stat_comp_benchmarks/tree/master/benchmarks/arma
"""

import os
import numpy as onp
import jax.config
import jax.numpy as np
import jax.lax as lax
import jax.api as api
import mlift
from mlift.transforms import normal_to_half_cauchy, normal_to_uniform
from experiments import common

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


dim_u = 4


def generate_params(u):
    return {
        "μ": u[0] * 10,
        "ϕ": normal_to_uniform(u[1]) * 2 - 1,
        "θ": normal_to_uniform(u[2]) * 2 - 1,
        "σ": normal_to_half_cauchy(u[3]) * 2.5,
    }


def generate_x(u, data):
    params = generate_params(u)

    def step(x, y):
        x = params["μ"] + params["ϕ"] * y + params["θ"] * (y - x)
        return x, x

    x_0 = params["μ"] * (1 + params["ϕ"])
    _, x_ = lax.scan(step, x_0, data["y_obs"][:-1])

    x = np.concatenate((np.array([x_0]), x_))

    return params, x


def generate_from_model(u, n, data):
    params, x = generate_x(u, data)
    y = x + params["σ"] * n
    return params, x, y


def generate_y(u, n, data):
    _, _, y = generate_from_model(u, n, data)
    return y


def posterior_neg_log_dens(u, data):
    params, x = generate_x(u, data)
    return (
        ((data["y_obs"] - x) / params["σ"]) ** 2 / 2 + np.log(params["σ"])
    ).sum() + (u ** 2).sum() / 2


def sample_initial_states(rng, args, data):
    """Sample initial states from prior."""
    init_states = []
    for _ in range(args.num_chain):
        u = rng.standard_normal(dim_u)
        if args.algorithm == "chmc":
            params, x = generate_x(u, data)
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
        "Run autoregressive moving average (ARMA) benchmark model experiment"
    )
    args = parser.parse_args()

    # Load data

    data = dict(np.load(os.path.join(args.data_dir, "arma-benchmark-data.npz")))
    dim_y = data["y_obs"].shape[0]

    # Set up seeded random number generator

    rng = onp.random.default_rng(args.seed)

    # Define variables to be traced

    def trace_func(state):
        u = state.pos[:dim_u]
        params = generate_params(u)
        return {**params, "u": u}

    # Run experiment

    final_states, traces, stats, summary_dict = common.run_experiment(
        args=args,
        data=data,
        dim_u=dim_u,
        rng=rng,
        experiment_name="arma",
        param_names=["μ", "ϕ", "θ", "σ"],
        param_trace_func=trace_func,
        posterior_neg_log_dens=posterior_neg_log_dens,
        constrained_system_class=mlift.IndependentAdditiveNoiseModelSystem,
        constrained_system_kwargs={
            "generate_y": generate_y,
            "data": data,
            "dim_u": dim_u,
        },
        sample_initial_states=sample_initial_states,
    )

