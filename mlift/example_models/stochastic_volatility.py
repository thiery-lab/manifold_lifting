"""Stochastic volatility time series model"""

import os
import numpy as onp
import jax
import jax.config
import jax.numpy as np
from mlift import construct_state_space_model_generators
from mlift.distributions import normal, half_normal, uniform
from mlift.prior import PriorSpecification, set_up_prior
import mlift.example_models.utils as utils


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


prior_specifications = {
    "μ": PriorSpecification(distribution=normal(0, 1)),
    "σ": PriorSpecification(distribution=half_normal(1)),
    "ϕ": PriorSpecification(distribution=uniform(-1, 1)),
}

compute_dim_u, generate_params, prior_neg_log_dens, sample_from_prior = set_up_prior(
    prior_specifications
)


def generate_x_0(params, v_0, data):
    return params["μ"] + (params["σ"] / (1 - params["ϕ"] ** 2) ** 0.5) * v_0


def forward_func(params, v, x, data):
    return params["μ"] + params["ϕ"] * (x - params["μ"]) + params["σ"] * v


def observation_func(params, n, x, data):
    return np.exp(x / 2) * n


def inverse_observation_func(params, n, y, data):
    return 2 * np.log(y / n)


generate_from_model, generate_y = construct_state_space_model_generators(
    generate_params=generate_params,
    generate_x_0=generate_x_0,
    forward_func=forward_func,
    observation_func=observation_func,
)


def extended_prior_neg_log_dens(q, data):
    dim_u = compute_dim_u(data)
    dim_y = data["y_obs"].shape[0]
    u, v, n = q[:dim_u], q[dim_u : dim_u + dim_y], q[dim_u + dim_y :]
    return prior_neg_log_dens(u, data) + (v ** 2).sum() / 2 + (n ** 2).sum() / 2


def posterior_neg_log_dens(q, data):
    dim_u = compute_dim_u(data)
    u, v = q[:dim_u], q[dim_u:]
    _, x = generate_from_model(u, v, data)
    return (
        prior_neg_log_dens(u, data)
        + (v ** 2).sum() / 2
        + (0.5 * ((data["y_obs"] / np.exp(x / 2)) ** 2).sum() + (x / 2).sum())
    )


def constr_split(u, v, n, y, data):
    params = generate_params(u, data)
    x = 2 * np.log(y / n)
    return (
        np.concatenate(
            (
                (
                    params["μ"]
                    + (params["σ"] / (1 - params["ϕ"] ** 2) ** 0.5) * v[0]
                    - x[0]
                )[None],
                params["μ"]
                + params["ϕ"] * (x[:-1] - params["μ"])
                + params["σ"] * v[1:]
                - x[1:],
            )
        ),
        x,
    )


def jacob_constr_split_blocks(u, v, n, y, data):
    dim_u = compute_dim_u(data)
    dim_y = y.shape[0]
    params, dparams_du = jax.jvp(
        lambda u_: generate_params(u_, data), (u,), (np.ones(dim_u),)
    )
    x = 2 * np.log(y / n)
    dx_dy = 2 / y
    one_minus_ϕ_sq = 1 - params["ϕ"] ** 2
    sqrt_one_minus_ϕ_sq = one_minus_ϕ_sq ** 0.5
    v_0_over_sqrt_one_minus_ϕ_sq = v[0] / sqrt_one_minus_ϕ_sq
    x_minus_μ = x[:-1] - params["μ"]
    dc_du = np.stack(
        (
            np.concatenate(
                (
                    dparams_du["μ"][None],
                    (1 - params["ϕ"]) * dparams_du["μ"] * np.ones(dim_y - 1),
                )
            ),
            np.concatenate(
                (
                    dparams_du["σ"][None] * v_0_over_sqrt_one_minus_ϕ_sq,
                    dparams_du["σ"] * v[1:],
                )
            ),
            np.concatenate(
                (
                    dparams_du["ϕ"][None]
                    * params["ϕ"]
                    * params["σ"]
                    * v_0_over_sqrt_one_minus_ϕ_sq
                    / one_minus_ϕ_sq,
                    x_minus_μ * dparams_du["ϕ"],
                )
            ),
        ),
        1,
    )
    dc_dv = np.concatenate(
        [params["σ"][None] / sqrt_one_minus_ϕ_sq, params["σ"] * np.ones(dim_y - 1)]
    )
    dc_dn = 2 / n, -2 * params["ϕ"] / n[:-1]
    c = np.concatenate(
        (
            (params["μ"] + params["σ"] * v_0_over_sqrt_one_minus_ϕ_sq - x[0])[None],
            params["μ"] + params["ϕ"] * x_minus_μ + params["σ"] * v[1:] - x[1:],
        )
    )
    return (dc_du, dc_dv, dc_dn, dx_dy), c


def sample_initial_states(rng, data, num_chain=4, algorithm="chmc"):
    """Sample initial states from prior."""
    init_states = []
    dim_y = data["y_obs"].shape[0]
    for _ in range(num_chain):
        u = sample_from_prior(rng, data)
        v = rng.standard_normal(dim_y)
        if algorithm == "chmc":
            _, x = generate_from_model(u, v, data)
            n = data["y_obs"] / onp.exp(x / 2)
            q = onp.concatenate((u, v, onp.asarray(n)))
        else:
            q = onp.concatenate((u, v))
        init_states.append(q)
    return init_states


if __name__ == "__main__":

    # Process command line arguments defining experiment parameters

    parser = utils.set_up_argparser_with_standard_arguments(
        "Run stochastic-volatility model simulated data experiment"
    )
    utils.add_ssm_specific_args(parser)
    args = parser.parse_args()

    # Load data

    data = dict(np.load(os.path.join(args.data_dir, "sv-simulated-data.npz")))
    dim_u = compute_dim_u(data)
    dim_y = data["y_obs"].shape[0]

    # Set up seeded random number generator

    rng = onp.random.default_rng(args.seed)

    # Define variables to be traced

    trace_func = utils.construct_trace_func(generate_params, data, dim_u, dim_v=dim_y)

    # Run experiment

    (
        constrained_system_class,
        constrained_system_kwargs,
    ) = utils.get_ssm_constrained_system_class_and_kwargs(
        args.use_manual_constraint_and_jacobian,
        generate_params,
        generate_x_0,
        forward_func,
        inverse_observation_func,
        constr_split,
        jacob_constr_split_blocks,
    )
    constrained_system_kwargs.update(data=data, dim_u=dim_u)

    final_states, traces, stats, summary_dict, sampler = utils.run_experiment(
        args=args,
        data=data,
        rng=rng,
        experiment_name="stochastic_volatility",
        var_names=list(prior_specifications.keys()),
        var_trace_func=trace_func,
        posterior_neg_log_dens=posterior_neg_log_dens,
        extended_prior_neg_log_dens=extended_prior_neg_log_dens,
        constrained_system_class=constrained_system_class,
        constrained_system_kwargs=constrained_system_kwargs,
        sample_initial_states=sample_initial_states,
    )

