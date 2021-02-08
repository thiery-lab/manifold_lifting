"""Stochastic volatility time series model"""

import os
import numpy as onp
import jax.config
import jax.numpy as np
import jax.lax as lax
import jax.api as api
import mlift
from mlift.transforms import normal_to_half_normal, normal_to_uniform
from experiments import common

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


dim_u = 3


def generate_params(u):
    return {
        "μ": u[0],
        "σ": normal_to_half_normal(u[1]),
        "ϕ": normal_to_uniform(u[2]) * 2 - 1,
    }


def generate_from_model(u, v):
    params = generate_params(u)
    x_0 = params["μ"] + (params["σ"] / (1 - params["ϕ"] ** 2) ** 0.5) * v[0]

    def step(x, v):
        x_ = params["μ"] + params["ϕ"] * (x - params["μ"]) + params["σ"] * v
        return x_, x_

    _, x_ = lax.scan(step, x_0, v[1:])

    return params, np.concatenate((x_0[None], x_))


def generate_y(u, v, n):
    _, x = generate_from_model(u, v)
    y = np.exp(x / 2) * n
    return y


def constr(u, v, n, data):
    params = generate_params(u)
    x = 2 * np.log(data["y_obs"] / n)
    return np.concatenate(
        (
            (params["μ"] + (params["σ"] / (1 - params["ϕ"] ** 2) ** 0.5) * v[0] - x[0])[
                None
            ],
            params["μ"]
            + params["ϕ"] * (x[:-1] - params["μ"])
            + params["σ"] * v[1:]
            - x[1:],
        )
    )


def jacob_constr_blocks(u, v, n, data):
    dim_y = data["y_obs"].shape[0]
    params, dparams_du = api.jvp(generate_params, (u,), (np.ones(dim_u),))
    x = 2 * np.log(data["y_obs"] / n)
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
    return (dc_du, dc_dv, dc_dn), c


def posterior_neg_log_dens(q, data):
    u, v = q[:dim_u], q[dim_u:]
    _, x = generate_from_model(u, v)
    return (
        0.5 * ((data["y_obs"] / np.exp(x / 2)) ** 2).sum()
        + (x / 2).sum()
        + (q ** 2).sum() / 2
    )


def sample_initial_states(rng, args, data):
    """Sample initial states from prior."""
    init_states = []
    dim_y = data["y_obs"].shape[0]
    for _ in range(args.num_chain):
        u = rng.standard_normal(dim_u)
        v = rng.standard_normal(dim_y)
        if args.algorithm == "chmc":
            _, x = generate_from_model(u, v)
            n = data["y_obs"] / onp.exp(x / 2)
            q = onp.concatenate((u, v, onp.asarray(n)))
            assert (
                abs(onp.exp(x / 2) * n - data["y_obs"]).max()
                < args.projection_solver_warm_up_constraint_tol
            )
        else:
            q = onp.concatenate((u, v))
        init_states.append(q)
    return init_states


if __name__ == "__main__":

    # Process command line arguments defining experiment parameters

    parser = common.set_up_argparser_with_standard_arguments(
        "Run stochastic-volatility model simulated data experiment"
    )
    args = parser.parse_args()

    # Load data

    data = dict(np.load(os.path.join(args.data_dir, "sv-simulated-data.npz")))
    dim_y = data["y_obs"].shape[0]

    # Set up seeded random number generator

    rng = onp.random.default_rng(args.seed)

    # Define variables to be traced

    def trace_func(state):
        u, v = state.pos[:dim_u], state.pos[dim_u : dim_u + dim_y]
        params = generate_params(u)
        return {**params,  "u": u, "v": v}

    # Run experiment

    final_states, traces, stats, summary_dict = common.run_experiment(
        args=args,
        data=data,
        dim_u=dim_u,
        rng=rng,
        experiment_name="sv",
        var_names=["μ", "σ", "ϕ"],
        var_trace_func=trace_func,
        posterior_neg_log_dens=posterior_neg_log_dens,
        constrained_system_class=mlift.PartiallyInvertibleStateSpaceModelSystem,
        constrained_system_kwargs={
            "constr_split": constr,
            "jacob_constr_blocks_split": jacob_constr_blocks,
            "data": data,
            "dim_u": dim_u

        },
        sample_initial_states=sample_initial_states,
    )

