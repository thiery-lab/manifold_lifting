"""Latent generalised autoregressive conditional heteroscedastic (GARCH) model."""

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


dim_u = 3


def generate_params(u):
    return {
        "α": normal_to_half_normal(u[0]),
        "β": normal_to_uniform(u[1]),
        "σ": normal_to_half_normal(u[2]),
    }


def generate_from_model(u, v):
    params = generate_params(u)
    x_0 = v[0]

    def step(x, v):
        x_ = np.sqrt(params["α"] + params["β"] * x ** 2) * v
        return x_, x_

    _, x_ = lax.scan(step, x_0, v[1:])

    x = np.concatenate((np.array([x_0]), x_))

    return params, x


def generate_y(u, v, n, data):
    params, x = generate_from_model(u, v)
    y = x + params["σ"] * n
    return y


def constr(u, v, n, data):
    params = generate_params(u)
    x = data["y_obs"] - params["σ"] * n
    return np.concatenate(
        (
            (v[0] - x[0])[None],
            np.sqrt(params["α"] + params["β"] * x[:-1] ** 2) * v[1:] - x[1:],
        )
    )


def jacob_constr_blocks(u, v, n, data):
    params, dparams_du = api.jvp(generate_params, (u,), (np.ones(dim_u),))
    dim_y = data["y_obs"].shape[0]
    x = data["y_obs"] - params["σ"] * n
    h = np.sqrt(params["α"] + params["β"] * x[:-1] ** 2)
    dc_du = np.stack(
        (
            np.concatenate((np.zeros(1), dparams_du["α"] * v[1:] / (2 * h))),
            np.concatenate(
                (np.zeros(1), dparams_du["β"] * x[:-1] ** 2 * v[1:] / (2 * h))
            ),
            np.concatenate(
                (
                    dparams_du["σ"][None] * n[0],
                    dparams_du["σ"]
                    * (n[1:] - params["β"] * n[:-1] * x[:-1] * v[1:] / h),
                )
            ),
        ),
        1,
    )
    dc_dv = np.concatenate((np.ones(1), h))
    dc_dn = (
        params["σ"] * np.ones(dim_y),
        -params["β"] * params["σ"] * x[:-1] * v[1:] / h,
    )
    c = np.concatenate((np.array([v[0] - x[0]]), h * v[1:] - x[1:]))
    return (dc_du, dc_dv, dc_dn), c


def posterior_neg_log_dens(q, data):
    u, v = q[:dim_u], q[dim_u:]
    params, x = generate_from_model(u, v)
    return (
        ((data["y_obs"] - x) / params["σ"]) ** 2 / 2 + np.log(params["σ"])
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
            n = (data["y_obs"] - x) / params["σ"]
            q = onp.concatenate((u, v, onp.asarray(n)))
            assert (
                abs(x + params["σ"] * n - data["y_obs"]).max()
                < args.projection_solver_warm_up_constraint_tol
            )
        else:
            q = onp.concatenate((u, v))
        init_states.append(q)
    return init_states


if __name__ == "__main__":

    # Process command line arguments defining experiment parameters

    parser = common.set_up_argparser_with_standard_arguments(
        "Run latent GARCH simulated data experiment"
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
        onp.load(os.path.join(args.data_dir, "latent-garch-simulated-data.npz"))
    )
    data["y_obs"] = data["x_obs"] + args.obs_noise_std * data["n_obs"]
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
        experiment_name="latent_garch",
        dir_prefix=f"σ_{args.obs_noise_std:.0e}",
        var_names=["α", "β", "σ"],
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

