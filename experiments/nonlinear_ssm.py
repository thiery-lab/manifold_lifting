"""Non-linearly observed state space model."""

import os
import numpy as onp
import jax.config
import jax.numpy as np
import jax.lax as lax
import jax.api as api
from mlift.transforms import normal_to_half_normal, normal_to_uniform
import mlift
from experiments import common

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


dim_u = 4


def generate_params(u):
    return {
        "μ": u[0],
        "γ": normal_to_half_normal(u[1]),
        "ρ": normal_to_uniform(u[2]) * 2 - 1,
        "σ": normal_to_half_normal(u[3]),
    }


def generate_from_model(u, v):
    params = generate_params(u)
    x_0 = params["μ"] + (params["γ"] / (1 - params["ρ"] ** 2) ** 0.5) * v[0]

    def step(x, v):
        x_ = params["μ"] + params["ρ"] * (x - params["μ"]) + params["γ"] * v
        return x_, x_

    _, x_ = lax.scan(step, x_0, v[1:])

    return params, np.concatenate((x_0[None], x_))


def generate_y(u, v, n):
    params, x = generate_from_model(u, v)
    y = np.exp(x) + params["σ"] * n
    return y


def constr(u, v, n, data):
    params = generate_params(u)
    x = np.log(data["y_obs"] - params["σ"] * n)
    return np.concatenate(
        (
            (params["μ"] + (params["γ"] / (1 - params["ρ"] ** 2) ** 0.5) * v[0] - x[0])[
                None
            ],
            params["μ"]
            + params["ρ"] * (x[:-1] - params["μ"])
            + params["γ"] * v[1:]
            - x[1:],
        )
    )


def jacob_constr_blocks(u, v, n, data):
    dim_y = data["y_obs"].shape[0]
    params, dparams_du = api.jvp(generate_params, (u,), (np.ones(dim_u),))
    exp_x = data["y_obs"] - params["σ"] * n
    x = np.log(exp_x)
    one_minus_ρ_sq = 1 - params["ρ"] ** 2
    sqrt_one_minus_ρ_sq = one_minus_ρ_sq ** 0.5
    v_0_over_sqrt_one_minus_ρ_sq = v[0] / sqrt_one_minus_ρ_sq
    x_minus_μ = x[:-1] - params["μ"]
    dc_du = np.stack(
        (
            np.concatenate(
                (
                    dparams_du["μ"][None],
                    (1 - params["ρ"]) * dparams_du["μ"] * np.ones(dim_y - 1),
                )
            ),
            np.concatenate(
                (
                    dparams_du["γ"][None] * v_0_over_sqrt_one_minus_ρ_sq,
                    dparams_du["γ"] * v[1:],
                )
            ),
            np.concatenate(
                (
                    dparams_du["ρ"][None]
                    * params["ρ"]
                    * params["γ"]
                    * v_0_over_sqrt_one_minus_ρ_sq
                    / one_minus_ρ_sq,
                    dparams_du["ρ"] * x_minus_μ,
                )
            ),
            np.concatenate(
                (
                    dparams_du["σ"][None] * n[0] / exp_x[0],
                    dparams_du["σ"]
                    * (n[1:] / exp_x[1:] - params["ρ"] * n[:-1] / exp_x[:-1]),
                )
            ),
        ),
        1,
    )
    dc_dv = np.concatenate(
        [params["γ"][None] / sqrt_one_minus_ρ_sq, params["γ"] * np.ones(dim_y - 1)]
    )
    dc_dn = params["σ"] / exp_x, -params["ρ"] * params["σ"] / exp_x[:-1]
    c = np.concatenate(
        (
            (params["μ"] + params["γ"] * v_0_over_sqrt_one_minus_ρ_sq - x[0])[None],
            params["μ"] + params["ρ"] * x_minus_μ + params["γ"] * v[1:] - x[1:],
        )
    )
    return (dc_du, dc_dv, dc_dn), c


def posterior_neg_log_dens(q, data):
    u, v = q[:dim_u], q[dim_u:]
    params, x = generate_from_model(u, v)
    return (
        ((data["y_obs"] - np.exp(x)) / params["σ"]) ** 2 / 2 + np.log(params["σ"])
    ).sum() + (q ** 2).sum() / 2


def sample_initial_states(rng, args, data):
    """Sample initial states from prior."""
    init_states = []
    dim_y = data["y_obs"].shape[0]
    for _ in range(args.num_chain):
        u = rng.standard_normal(dim_u)
        v = rng.standard_normal(dim_y)
        if args.algorithm == "chmc":
            params, x = generate_from_model(u, v)
            n = (data["y_obs"] - onp.exp(x)) / params["σ"]
            q = onp.concatenate((u, v, onp.asarray(n)))
            assert (
                abs(onp.exp(x) + params["σ"] * n - data["y_obs"]).max()
                < args.projection_solver_warm_up_constraint_tol
            )
        else:
            q = onp.concatenate((u, v))
        init_states.append(q)
    return init_states


if __name__ == "__main__":

    # Process command line arguments defining experiment parameters

    parser = common.set_up_argparser_with_standard_arguments(
        "Run nonlinear state space model simulated data experiment"
    )
    parser.add_argument(
        "--obs-noise-std",
        type=float,
        default=1.0,
        help="Standard deviation of observation noise to use in simulated data",
    )
    args = parser.parse_args()

    # Load data

    data = dict(
        onp.load(os.path.join(args.data_dir, "nonlinear-ssm-simulated-data.npz"))
    )
    data["y_obs"] = onp.exp(data["x_obs"]) + args.obs_noise_std * data["n_obs"]
    dim_y = data["y_obs"].shape[0]

    # Set up seeded random number generator

    rng = onp.random.default_rng(args.seed)

    # Define variables to be traced

    def trace_func(state):
        u, v = state.pos[:dim_u], state.pos[dim_u : dim_u + dim_y]
        params = generate_params(u)
        return {**params, "u": u, "v": v}

    # Run experiment

    final_states, traces, stats, summary_dict = common.run_experiment(
        args=args,
        data=data,
        dim_u=dim_u,
        rng=rng,
        experiment_name="nonlinear_ssm",
        dir_prefix=f"σ_{args.obs_noise_std:.0e}",
        var_names=["μ", "ρ", "γ", "σ"],
        var_trace_func=trace_func,
        posterior_neg_log_dens=posterior_neg_log_dens,
        constrained_system_class=mlift.PartiallyInvertibleStateSpaceModelSystem,
        constrained_system_kwargs={
            "constr_split": constr,
            "jacob_constr_blocks_split": jacob_constr_blocks,
            "data": data,
            "dim_u": dim_u,
        },
        sample_initial_states=sample_initial_states,
    )

