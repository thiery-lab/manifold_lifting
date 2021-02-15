"""Generalised autoregressive conditional heteroscedastic (GARCH) benchmark model

Model definition and data taken from:

https://github.com/stan-dev/stat_comp_benchmarks/tree/master/benchmarks/garch
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
        "α_0": normal_to_half_cauchy(u[1]) * 2.5,
        "α_1": normal_to_uniform(u[2]),
        "β_1": (1 - normal_to_uniform(u[2])) * normal_to_uniform(u[3]),
    }


def generate_from_model(u, data):
    params = generate_params(u)

    def step(x, y):
        x = params["α_0"] + params["α_1"] * (y - params["μ"]) ** 2 + params["β_1"] * x
        return x, x

    _, x_ = lax.scan(step, data["x_0"], data["y_obs"][:-1])

    x = np.concatenate((np.array([data["x_0"]]), x_))

    return params, x


def generate_y(u, n, data):
    params, x = generate_from_model(u, data)
    y = params["μ"] + np.sqrt(x) * n
    return y


def posterior_neg_log_dens(u, data):
    params, x = generate_from_model(u, data)
    return (
        ((data["y_obs"] - params["μ"]) ** 2 / x).sum() / 2
        + np.log(x).sum() / 2
        + (u ** 2).sum() / 2
    )


def sample_initial_states(rng, args, data):
    """Sample initial states from prior."""
    init_states = []
    for _ in range(args.num_chain):
        u = rng.standard_normal(dim_u)
        if args.algorithm == "chmc":
            params, x = generate_from_model(u, data)
            n = (data["y_obs"] - params["μ"]) / onp.sqrt(x)
            q = onp.concatenate((u, onp.asarray(n)))
            assert (
                abs(params["μ"] + np.sqrt(x) * n - data["y_obs"]).max()
                < args.projection_solver_warm_up_constraint_tol
            )
        else:
            q = u
        init_states.append(q)
    return init_states


if __name__ == "__main__":

    # Process command line arguments defining experiment parameters

    parser = common.set_up_argparser_with_standard_arguments(
        "Run generalised autoregressive conditional heteroscedasticity (GARCH) "
        "benchmark model experiment"
    )
    args = parser.parse_args()

    # Load data

    data = dict(np.load(os.path.join(args.data_dir, "garch-benchmark-data.npz")))

    # Set up seeded random number generator

    rng = onp.random.default_rng(args.seed)

    # Define variables to be traced

    def trace_func(state):
        u = state.pos[:dim_u]
        params = generate_params(u)
        return {**params, "u": u}

    # Run experiment

    final_states, traces, stats, summary_dict, sampler = common.run_experiment(
        args=args,
        data=data,
        dim_u=dim_u,
        rng=rng,
        experiment_name="garch",
        var_names=["μ", "α_0", "α_1", "β_1"],
        var_trace_func=trace_func,
        posterior_neg_log_dens=posterior_neg_log_dens,
        constrained_system_class=mlift.IndependentAdditiveNoiseModelSystem,
        constrained_system_kwargs={
            "generate_y": generate_y,
            "data": data,
            "dim_u": dim_u,
        },
        sample_initial_states=sample_initial_states,
    )

