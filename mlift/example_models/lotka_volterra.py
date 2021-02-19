"""Lotka-Volterra predator-prey model.

Source: https://github.com/stan-dev/example-models/
"""

import os
import numpy as onp
import jax.config
import jax.numpy as np
import jax.lax as lax
import jax.api as api
from mlift.systems import IndependentAdditiveNoiseModelSystem
from mlift.distributions import truncated_normal, log_normal
from mlift.ode import integrate_ode_rk4
from mlift.prior import PriorSpecification, set_up_prior
import mlift.example_models.utils as utils

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


prior_specifications = {
    "α": PriorSpecification(distribution=truncated_normal(1, 0.5, 0)),
    "β": PriorSpecification(distribution=truncated_normal(0.05, 0.05, 0)),
    "γ": PriorSpecification(distribution=truncated_normal(1, 0.5, 0)),
    "δ": PriorSpecification(distribution=truncated_normal(0.05, 0.05, 0)),
    "σ": PriorSpecification(shape=(2,), distribution=log_normal(-1, 1)),
    "x_init": PriorSpecification(shape=(2,), distribution=log_normal(np.log(10), 1)),
}


compute_dim_u, generate_params, prior_neg_log_dens, sample_from_prior = set_up_prior(
    prior_specifications
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
    params = generate_params(u, data)
    x_seq = integrate_ode_rk4(
        dx_dt, params["x_init"], data["t_seq"], params, data["dt"]
    )
    return params, x_seq


def generate_y(u, n, data):
    params, x = generate_from_model(u, data)
    return observation_func(x) + np.tile(params["σ"], x.shape[0]) * n


def extended_prior_neg_log_dens(q, data):
    dim_u = compute_dim_u(data)
    u, n = q[:dim_u], q[dim_u:]
    return prior_neg_log_dens(u, data) + (n ** 2).sum() / 2


def posterior_neg_log_dens(u, data):
    params, x = generate_from_model(u, data)
    y_mean = observation_func(x)
    σ_rep = np.tile(params["σ"], x.shape[0])
    return (
        prior_neg_log_dens(u, data)
        + (((y_mean - data["y_obs"]) / σ_rep) ** 2 / 2 + np.log(σ_rep)).sum()
    )


def sample_initial_states(rng, args, data):
    """Sample initial state using approximate Bayesian computation reject type approach.

    Use an approximate Bayesian computation type approach of repeatedly sampling from
    prior until peak times of sequence (without noise) generated from state matches peak
    times of (noisy) observation sequence to within a tolerance. This helps to avoid
    chains getting trapped in 'bad' modes.
    """
    init_states = []
    x_obs = np.exp(data["y_obs"]).reshape((-1, 2))
    peak_times_obs_0 = utils.calculate_peak_times(x_obs[:, 0], data["t_seq"], 1, 10)
    peak_times_obs_1 = utils.calculate_peak_times(x_obs[:, 1], data["t_seq"], 1, 10)
    jitted_generate_from_model = api.jit(api.partial(generate_from_model, data=data))
    num_tries = 0
    while len(init_states) < args.num_chain and num_tries < args.max_init_tries:
        u = sample_from_prior(rng, data)
        params, x = jitted_generate_from_model(u)
        if not onp.all(onp.isfinite(x)) or not onp.all(x > 0):
            num_tries += 1
            continue
        peak_times_0 = utils.calculate_peak_times(x[:, 0], data["t_seq"], 1, 10)
        peak_times_1 = utils.calculate_peak_times(x[:, 1], data["t_seq"], 1, 10)
        if not (
            peak_times_0.shape[0] == peak_times_obs_0.shape[0]
            and peak_times_1.shape[0] == peak_times_obs_1.shape[0]
            and abs(peak_times_0 - peak_times_obs_0).max()
            <= args.init_peak_time_diff_threshold
            and abs(peak_times_1 - peak_times_obs_1).max()
            <= args.init_peak_time_diff_threshold
        ):
            num_tries += 1
            continue
        if args.algorithm == "chmc":
            y_mean = observation_func(x)
            σ_rep = onp.tile(params["σ"], x.shape[0])
            n = (data["y_obs"] - y_mean) / σ_rep
            q = onp.concatenate((u, onp.asarray(n)))
            assert (
                abs(y_mean + σ_rep * n - data["y_obs"]).max()
                < args.projection_solver_warm_up_constraint_tol
            )
        else:
            q = u
        init_states.append(q)
        num_tries += 1
    if len(init_states) != args.num_chain:
        raise RuntimeError(
            f"Failed to find {args.num_chain} acceptable initial states in "
            f"{args.max_init_tries} tries."
        )
    return init_states


if __name__ == "__main__":

    # Process command line arguments defining experiment parameters

    parser = utils.set_up_argparser_with_standard_arguments(
        "Run Lotka-Volterra model Hudson-Lynx data experiment"
    )
    parser.add_argument(
        "--obs-noise-std",
        type=float,
        default=1.0,
        help="Standard deviation of observation noise to use in simulated data",
    )
    parser.add_argument(
        "--max-init-tries",
        type=int,
        default=1000,
        help="Maximum number of prior samples to try to find acceptable initial states",
    )
    parser.add_argument(
        "--init-peak-time-diff-threshold",
        type=float,
        default=2.0,
        help="Maximum difference between peak times of initial state and observations",
    )
    args = parser.parse_args()

    # Load data

    data = dict(onp.load(os.path.join(args.data_dir, "lotka-volterra-hudson-data.npz")))
    dim_u = compute_dim_u(data)

    # Set up seeded random number generator

    rng = onp.random.default_rng(args.seed)

    # Define variables to be traced

    trace_func = utils.construct_trace_func(generate_params, data, dim_u)

    # Run experiment

    final_states, traces, stats, summary_dict, sampler = utils.run_experiment(
        args=args,
        data=data,
        dim_u=dim_u,
        rng=rng,
        experiment_name="lotka_volterra",
        var_names=list(prior_specifications.keys()),
        var_trace_func=trace_func,
        posterior_neg_log_dens=posterior_neg_log_dens,
        extended_prior_neg_log_dens=extended_prior_neg_log_dens,
        constrained_system_class=IndependentAdditiveNoiseModelSystem,
        constrained_system_kwargs={
            "generate_y": generate_y,
            "data": data,
            "dim_u": dim_u,
        },
        sample_initial_states=sample_initial_states,
    )

