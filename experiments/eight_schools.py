"""Hierarchical model for 'eight-schools' data (Rubin, 1981).

Model definition and data taken from:

https://github.com/stan-dev/stat_comp_benchmarks/tree/master/benchmarks/eight_schools

References:

    Rubin, D. (1981). Estimation in Parallel Randomized Experiments.
    Journal of Educational Statistics, 6(4), 377-401. doi:10.2307/1164617
"""

import os
import numpy as onp
import jax.config
import jax.numpy as np
import jax.lax as lax
import jax.api as api
import mlift
from mlift.transforms import normal_to_half_cauchy
from experiments import common

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


dim_u = 2


def generate_params(u):
    return {
        "μ": u[0] * 5,
        "τ": normal_to_half_cauchy(u[1]) * 5,
    }


def generate_x(u, v):
    params = generate_params(u)
    x = params["μ"] + params["τ"] * v
    return params, x


def generate_from_model(u, v, n, data):
    params, x = generate_x(u, v)
    y = x + data["σ"] * n
    return params, x, y


def generate_y(u, v, n, data):
    _, _, y = generate_from_model(u, v, n, data)
    return y


def posterior_neg_log_dens(q, data):
    u, v = q[:dim_u], q[dim_u:]
    _, x = generate_x(u, v)
    return (((data["y_obs"] - x) / data["σ"]) ** 2 / 2 + np.log(data["σ"])).sum() + (
        q ** 2
    ).sum() / 2


def sample_initial_states(rng, args, data):
    """Sample initial states from prior."""
    init_states = []
    dim_y = data["y_obs"].shape[0]
    for _ in range(args.num_chain):
        u = rng.standard_normal(dim_u)
        v = rng.standard_normal(dim_y)
        if args.algorithm == "chmc":
            _, x = generate_x(u, v)
            n = (data["y_obs"] - x) / data["σ"]
            q = onp.concatenate((u, v, onp.asarray(n)))
            assert (
                abs(x + data["σ"] * n - data["y_obs"]).max()
                < args.projection_solver_warm_up_constraint_tol
            )
        else:
            q = onp.concatenate((u, v))
        init_states.append(q)
    return init_states


if __name__ == "__main__":

    # Process command line arguments defining experiment parameters

    parser = common.set_up_argparser_with_standard_arguments(
        "Run eight-school hierarchical model experiment"
    )
    args = parser.parse_args()

    # Load data

    data = dict(np.load(os.path.join(args.data_dir, "eight-schools-data.npz")))
    dim_y = data["y_obs"].shape[0]

    # Set up seeded random number generator

    rng = onp.random.default_rng(args.seed)

    # Define variables to be traced

    def trace_func(state):
        u, v = state.pos[:dim_u], state.pos[dim_u : dim_u + dim_y]
        params, x = generate_x(u, v)
        return {**params, "x": x, "u": u, "v": v}

    # Run experiment

    final_states, traces, stats, summary_dict = common.run_experiment(
        args=args,
        data=data,
        dim_u=dim_u,
        rng=rng,
        experiment_name="eight_schools",
        var_names=["μ", "τ", "x"],
        var_trace_func=trace_func,
        posterior_neg_log_dens=posterior_neg_log_dens,
        constrained_system_class=mlift.HierarchicalLatentVariableModelSystem,
        constrained_system_kwargs={
            "generate_y": generate_y,
            "data": data,
            "dim_u": dim_u,
        },
        sample_initial_states=sample_initial_states,
    )

