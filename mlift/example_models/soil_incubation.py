"""Two-pool linear model with feedback.

Source: https://github.com/stan-dev/example-models/
"""

import os
import numpy as onp
import jax.config
import jax.numpy as np
from mlift.systems import IndependentAdditiveNoiseModelSystem
from mlift.distributions import half_normal, half_cauchy, beta
from mlift.ode import integrate_ode_rk4
from mlift.prior import PriorSpecification, set_up_prior
import mlift.example_models.utils as utils

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


prior_specifications = {
    "k_1": PriorSpecification(distribution=half_normal(1)),
    "k_2": PriorSpecification(distribution=half_normal(1)),
    "α_21": PriorSpecification(distribution=half_normal(1)),
    "α_12": PriorSpecification(distribution=half_normal(1)),
    "γ": PriorSpecification(distribution=beta(10, 1)),
    "σ": PriorSpecification(distribution=half_cauchy(1)),
}

compute_dim_u, generate_params, prior_neg_log_dens, sample_from_prior = set_up_prior(
    prior_specifications
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
    params = generate_params(u, data)
    x_init = np.array(
        (
            params["γ"] * data["initial_total_carbon"],
            (1 - params["γ"]) * data["initial_total_carbon"],
        )
    )
    x = integrate_ode_rk4(dx_dt, x_init, data["t_seq"], params, data["dt"])
    return params, x


def generate_y(u, n, data):
    params, x = generate_from_model(u, data)
    return observation_func(x, data) + params["σ"] * n


def extended_prior_neg_log_dens(q, data):
    dim_u = compute_dim_u(data)
    u, n = q[:dim_u], q[dim_u:]
    return prior_neg_log_dens(u, data) + (n ** 2).sum() / 2


def posterior_neg_log_dens(u, data):
    params, x = generate_from_model(u, data)
    y_mean = observation_func(x, data)
    return (
        prior_neg_log_dens(u, data)
        + (
            ((y_mean - data["y_obs"]) / params["σ"]) ** 2 / 2 + np.log(params["σ"])
        ).sum()
    )


def sample_initial_states(
    rng, data, num_chain=4, algorithm="chmc", mean_residual_sq_threshold=1e6
):
    """Sample initial states from prior."""
    init_states = []
    while len(init_states) < num_chain:
        u = sample_from_prior(rng, data)
        params, x = generate_from_model(u, data)
        y_mean = observation_func(x, data)
        if not onp.all(np.isfinite(y_mean)):
            continue
        n = (data["y_obs"] - y_mean) / params["σ"]
        if (n ** 2).mean() > mean_residual_sq_threshold:
            # Skip initialisations with large residuals as can cause numerical issues
            continue
        if algorithm == "chmc":
            q = onp.concatenate((u, onp.asarray(n)))
        else:
            q = onp.asarray(u)
        init_states.append(q)
    return init_states


if __name__ == "__main__":

    # Process command line arguments defining experiment parameters

    parser = utils.set_up_argparser_with_standard_arguments(
        "Run soil-incubation model experiment"
    )
    args = parser.parse_args()

    # Load data

    data = dict(onp.load(os.path.join(args.data_dir, "soil-incubation-data.npz")))
    dim_u = compute_dim_u(data)

    # Set up seeded random number generator

    rng = onp.random.default_rng(args.seed)

    # Define variables to be traced

    trace_func = utils.construct_trace_func(generate_params, data, dim_u)

    # Run experiment

    final_states, traces, stats, summary_dict, sampler = utils.run_experiment(
        args=args,
        data=data,
        rng=rng,
        experiment_name="soil_incubation",
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

