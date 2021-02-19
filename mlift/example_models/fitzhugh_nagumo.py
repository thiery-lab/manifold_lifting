"""FitzHugh-Nagumo model of neuronal action potential generation."""

import os
from collections import namedtuple
import numpy as onp
import jax.config
import jax.numpy as np
import jax.lax as lax
import jax.api as api
from mlift.systems import IndependentAdditiveNoiseModelSystem
from mlift.distributions import log_normal, normal
from mlift.ode import integrate_ode_rk4
from mlift.prior import PriorSpecification, set_up_prior
import mlift.example_models.utils as utils

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


prior_specifications = {
    "α": PriorSpecification(distribution=log_normal(0, 1)),
    "β": PriorSpecification(distribution=log_normal(0, 1)),
    "γ": PriorSpecification(distribution=log_normal(0, 1)),
    "δ": PriorSpecification(distribution=log_normal(-1, 1)),
    "ϵ": PriorSpecification(distribution=log_normal(-3, 1)),
    "ζ": PriorSpecification(distribution=log_normal(-2, 1)),
    "σ": PriorSpecification(distribution=log_normal(-1, 1)),
    "x_init": PriorSpecification(shape=(2,), distribution=normal(0, 1)),
}

compute_dim_u, generate_params, prior_neg_log_dens, sample_from_prior = set_up_prior(
    prior_specifications
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
    params = generate_params(u, data)
    x_seq = integrate_ode_rk4(
        dx_dt, params["x_init"], data["t_seq"], params, data["dt"]
    )
    return params, x_seq


def generate_y(u, n, data):
    params, x = generate_from_model(u, data)
    return observation_func(x) + params["σ"] * n


def extended_prior_neg_log_dens(q, data):
    dim_u = compute_dim_u(data)
    u, n = q[:dim_u], q[dim_u:]
    return prior_neg_log_dens(u, data) + (n ** 2).sum() / 2


def posterior_neg_log_dens(u, data):
    params, x = generate_from_model(u, data)
    y_mean = observation_func(x)
    return (
        ((y_mean - data["y_obs"]) / params["σ"]) ** 2 / 2 + np.log(params["σ"])
    ).sum() + prior_neg_log_dens(u, data)


def sample_initial_states(rng, args, data):
    """Sample initial state using approximate Bayesian computation reject type approach.

    Use an approximate Bayesian computation type approach of repeatedly sampling from
    prior until peak times of sequence (without noise) generated from state matches peak
    times of (noisy) observation sequence to within a tolerance. This helps to avoid
    chains getting trapped in 'bad' modes.
    """
    init_states = []
    peak_times_obs = utils.calculate_peak_times(data["y_obs"], data["t_seq"], 40, 0)
    num_tries = 0
    jitted_generate_from_model = api.jit(api.partial(generate_from_model, data=data))
    while len(init_states) < args.num_chain and num_tries < args.max_init_tries:
        u = sample_from_prior(rng, data)
        params, x = jitted_generate_from_model(u)
        if not onp.all(onp.isfinite(x)):
            num_tries += 1
            continue
        y_mean = observation_func(x)
        peak_times = utils.calculate_peak_times(y_mean, data["t_seq"], 40, 0)
        if not (
            peak_times.shape[0] == peak_times_obs.shape[0]
            and abs(peak_times - peak_times_obs).max()
            < args.init_peak_time_diff_threshold
        ):
            num_tries += 1
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
        "Run FitzHuhgh-Nagumo model simulated data experiment"
    )
    parser.add_argument(
        "--obs-noise-std",
        type=float,
        default=0.1,
        help="Standard deviation of observation noise to use in simulated data",
    )
    parser.add_argument(
        "--max-init-tries",
        type=int,
        default=10000,
        help="Maximum number of prior samples to try to find acceptable initial states",
    )
    parser.add_argument(
        "--init-peak-time-diff-threshold",
        type=float,
        default=1.0,
        help="Maximum difference between peak times of initial state and observations",
    )
    args = parser.parse_args()

    # Load data

    data = dict(
        onp.load(os.path.join(args.data_dir, "fitzhugh-nagumo-simulated-data.npz"))
    )
    data["y_obs"] = data["y_mean_obs"] + args.obs_noise_std * data["n_obs"]
    dim_u = compute_dim_u(data)

    # Set up seeded random number generator

    rng = onp.random.default_rng(args.seed)

    # Define variables to be traced

    jitted_generate_params = api.jit(api.partial(generate_params, data=data))

    def trace_func(state):
        u = state.pos[:dim_u]
        params = jitted_generate_params(u)
        return {**params, "u": u}

    # Run experiment

    final_states, traces, stats, summary_dict, sampler = utils.run_experiment(
        args=args,
        data=data,
        dim_u=dim_u,
        rng=rng,
        experiment_name="fitzhugh_nagumo",
        dir_prefix=f"σ_{args.obs_noise_std:.0e}",
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

