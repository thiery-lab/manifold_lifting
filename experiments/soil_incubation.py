"""Two-pool linear model with feedback.

Source: https://github.com/stan-dev/example-models/
"""

import os
from collections import namedtuple
import numpy as onp
import jax.config
import jax.numpy as np
import jax.lax as lax
import jax.api as api
import mlift
from mlift.transforms import (
    normal_to_half_cauchy,
    normal_to_half_normal,
    normal_to_uniform,
)
from mlift.ode import integrate_ode_rk4
from experiments import common

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


def normal_to_beta_10_1(n):
    return normal_to_uniform(n) ** 0.1


param_prior_specifications = {
    "k_1": common.PriorSpecification(shape=(), transform=normal_to_half_normal),
    "k_2": common.PriorSpecification(shape=(), transform=normal_to_half_normal),
    "α_21": common.PriorSpecification(shape=(), transform=normal_to_half_normal),
    "α_12": common.PriorSpecification(shape=(), transform=normal_to_half_normal),
    "γ": common.PriorSpecification(shape=(), transform=normal_to_beta_10_1),
    "σ": common.PriorSpecification(shape=(), transform=normal_to_half_cauchy),
}

generate_params, dim_u = common.get_param_generator_and_dimension(
    param_prior_specifications
)


def dx_dt(x, t, params):
    return np.array(
        (
            -params["k_1"] * x[0] + params["α_12"] * params["k_2"] * x[1],
            -params["k_2"] * x[1] + params["α_21"] * params["k_1"] * x[0],
        )
    )


def observation_func(x, data):
    return data["initial_total_carbon"] - x[1:].sum(-1)


def generate_from_model(u, data):
    params = generate_params(u)
    x_init = np.array(
        (
            params["γ"] * data["initial_total_carbon"],
            (1 - params["γ"]) * data["initial_total_carbon"],
        )
    )
    x_seq = integrate_ode_rk4(dx_dt, x_init, data["t_seq"], params, data["dt"])
    return params, x_seq


def generate_y(u, n, data):
    params, x = generate_from_model(u, data)
    return observation_func(x, data) + params["σ"] * n


def posterior_neg_log_dens(u, data):
    params, x = generate_from_model(u, data)
    y_mean = observation_func(x, data)
    return (
        ((y_mean - data["y_obs"]) / params["σ"]) ** 2 / 2 + np.log(params["σ"])
    ).sum() + np.sum(u ** 2) / 2


def sample_initial_states(rng, args, data):
    """Sample initial states from prior."""
    init_states = []
    while len(init_states) < args.num_chain:
        u = rng.standard_normal(dim_u)
        params, x = generate_from_model(u, data)
        y_mean = observation_func(x, data)
        if not onp.all(np.isfinite(y_mean)):
            continue
        if args.algorithm == "chmc":
            n = (data["y_obs"] - y_mean) / params["σ"]
            q = onp.concatenate((u, onp.asarray(n)))
            assert (
                abs(y_mean + params["σ"] * n - data["y_obs"]).max()
                < args.projection_solver_warm_up_constraint_tol
            )
        else:
            q = u
        init_states.append(q)
    return init_states


if __name__ == "__main__":

    # Process command line arguments defining experiment parameters

    parser = common.set_up_argparser_with_standard_arguments(
        "Run soil-incubation model experiment"
    )
    args = parser.parse_args()

    # Load data

    data = dict(onp.load(os.path.join(args.data_dir, "soil-incubation-data.npz")))

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
        experiment_name="soil_incubation",
        var_names=list(param_prior_specifications.keys()),
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

