"""FitzHugh-Nagumo model of neuronal action potential generation."""

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
    "α": common.PriorSpecification(shape=(), transform=np.exp),
    "β": common.PriorSpecification(shape=(), transform=np.exp),
    "γ": common.PriorSpecification(shape=(), transform=np.exp),
    "δ": common.PriorSpecification(shape=(), transform=lambda u: np.exp(u - 1)),
    "ϵ": common.PriorSpecification(shape=(), transform=lambda u: np.exp(u - 2)),
    "ζ": common.PriorSpecification(shape=(), transform=lambda u: np.exp(u - 2)),
    "σ": common.PriorSpecification(shape=(), transform=lambda u: np.exp(u - 1)),
    "x_init": common.PriorSpecification(shape=(2,), transform=None),
}

generate_params, dim_u = common.get_param_generator_and_dimension(
    param_prior_specifications
)


def dx_dt(x, t, params):
    return np.array(
        (
            params["α"] * x[0] - params["β"] * x[0] ** 3 + params["γ"] * x[1],
            -params["δ"] * x[0] - params["ϵ"] * x[1] + params["ζ"],
        )
    )


def observation_func(x):
    return x[1:, 0]


def generate_from_model(u, data):
    params = generate_params(u)
    x_seq = integrate_ode_rk4(
        dx_dt, params["x_init"], data["t_seq"], params, data["dt"]
    )
    return params, x_seq


def generate_y(u, n, data):
    params, x = generate_from_model(u, data)
    return observation_func(x) + params["σ"] * n


def posterior_neg_log_dens(u, data):
    params, x = generate_from_model(u, data)
    y_mean = observation_func(x)
    return (
        ((y_mean - data["y_obs"]) / params["σ"]) ** 2 / 2 + np.log(params["σ"])
    ).sum() + np.sum(u ** 2) / 2


def sample_initial_states(rng, args, data):
    """Sample initial states from prior."""
    init_states = []
    for _ in range(args.num_chain):
        u = rng.standard_normal(dim_u)
        if args.algorithm == "chmc":
            params, x = generate_from_model(u, data)
            y_mean = observation_func(x)
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
        "Run FitzHuhgh-Nagumo model simulated data experiment"
    )
    parser.add_argument(
        "--obs-noise-std",
        type=float,
        default=0.1,
        help="Standard deviation of observation noise to use in simulated data",
    )
    args = parser.parse_args()

    # Load data

    data = dict(
        onp.load(os.path.join(args.data_dir, "fitzhugh-nagumo-simulated-data.npz"))
    )
    data["y_obs"] = data["y_mean_obs"] + args.obs_noise_std * data["n_obs"]

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
        experiment_name="fitzhugh_nagumo",
        dir_prefix=f"σ_{args.obs_noise_std:.0e}",
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

