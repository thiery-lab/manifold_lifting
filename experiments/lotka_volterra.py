"""Lotka-Volterra predator-prey model.

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
from mlift.ode import integrate_ode_rk4
from experiments import common

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


param_prior_specifications = {
    "α": common.PriorSpecification(shape=(), transform=lambda u: 0.5 * u + 1),
    "β": common.PriorSpecification(shape=(), transform=lambda u: 0.05 * u + 0.05),
    "γ": common.PriorSpecification(shape=(), transform=lambda u: 0.5 * u + 1),
    "δ": common.PriorSpecification(shape=(), transform=lambda u: 0.05 * u + 0.05),
    "σ": common.PriorSpecification(shape=(2,), transform=lambda u: np.exp(u - 1)),
    "x_init": common.PriorSpecification(
        shape=(2,), transform=lambda u: np.exp(u + np.log(10))
    ),
}

generate_params, dim_u = common.get_param_generator_and_dimension(
    param_prior_specifications
)


def dx_dt(x, t, params):
    return np.array(
        (
            (params["α"] - params["β"] * x[1]) * x[0],
            (-params["γ"] + params["δ"] * x[0]) * x[1],
        )
    )


def observation_func(x):
    return np.log(x).flatten()


def generate_from_model(u, data):
    params = generate_params(u)
    x_seq = integrate_ode_rk4(
        dx_dt, params["x_init"], data["t_seq"], params, data["dt"]
    )
    return params, x_seq


def generate_y(u, n, data):
    params, x = generate_from_model(u, data)
    return observation_func(x) + np.repeat(params["σ"], x.shape[0]) * n


def posterior_neg_log_dens(u, data):
    params, x = generate_from_model(u, data)
    y_mean = observation_func(x)
    σ_rep = np.repeat(params["σ"], x.shape[0])
    return (((y_mean - data["y_obs"]) / σ_rep) ** 2 / 2 + np.log(σ_rep)).sum() + np.sum(
        u ** 2
    ) / 2


def sample_initial_states(rng, args, data):
    """Sample initial states from prior."""
    init_states = []
    while len(init_states) < args.num_chain:
        u = rng.standard_normal(dim_u)
        params, x = generate_from_model(u, data)
        y_mean = observation_func(x)
        if not onp.all(np.isfinite(y_mean)):
            continue
        if args.algorithm == "chmc":
            σ_rep = onp.repeat(params["σ"], x.shape[0])
            n = (data["y_obs"] - y_mean) / σ_rep
            q = onp.concatenate((u, onp.asarray(n)))
            assert (
                abs(y_mean + σ_rep * n - data["y_obs"]).max()
                < args.projection_solver_warm_up_constraint_tol
            )
        else:
            q = u
        init_states.append(q)
    return init_states


if __name__ == "__main__":

    # Process command line arguments defining experiment parameters

    parser = common.set_up_argparser_with_standard_arguments(
        "Run Lotka-Volterra model Hudson-Lynx data experiment"
    )
    args = parser.parse_args()

    # Load data

    data = dict(onp.load(os.path.join(args.data_dir, "lotka-volterra-hudson-data.npz")))

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
        experiment_name="lotka_volterra",
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

