"""Latent generalised autoregressive conditional heteroscedastic (GARCH) model."""

import os
import numpy as onp
import jax.config
import jax.numpy as np
import jax.lax as lax
import jax.api as api
import mlift
from mlift.distributions import half_normal, uniform
from experiments import common

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


prior_specifications = {
    "α": common.PriorSpecification(distribution=half_normal(1)),
    "β": common.PriorSpecification(distribution=uniform(0, 1)),
    "σ": common.PriorSpecification(distribution=half_normal(1)),
}

(
    compute_dim_u,
    generate_params,
    prior_neg_log_dens,
    sample_from_prior,
) = common.set_up_prior(prior_specifications)


def generate_x_0(params, v_0, data):
    return v_0


def forward_func(params, v, x, data):
    return np.sqrt(params["α"] + params["β"] * x ** 2) * v


def observation_func(params, n, x, data):
    return x + params["σ"] * n


def inverse_observation_func(params, n, y, data):
    return y - params["σ"] * n


generate_from_model, generate_y = mlift.construct_state_space_model_generators(
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
    params, x = generate_from_model(u, v, data)
    return (
        prior_neg_log_dens(u, data)
        + (v ** 2).sum() / 2
        + (((data["y_obs"] - x) / params["σ"]) ** 2 / 2 + np.log(params["σ"])).sum()
    )


def constr_split(u, v, n, y, data):
    params = generate_params(u, data)
    x = y - params["σ"] * n
    return (
        np.concatenate(
            (
                (v[0] - x[0])[None],
                np.sqrt(params["α"] + params["β"] * x[:-1] ** 2) * v[1:] - x[1:],
            )
        ),
        x,
    )


def jacob_constr_split_blocks(u, v, n, y, data):
    params, dparams_du = api.jvp(
        lambda u_: generate_params(u_, data), (u,), (np.ones(dim_u),)
    )
    dim_y = y.shape[0]
    x = y - params["σ"] * n
    dx_dy = np.ones(dim_y)
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
    return (dc_du, dc_dv, dc_dn, dx_dy), c


def sample_initial_states(rng, args, data):
    """Sample initial states from prior."""
    init_states = []
    dim_y = data["y_obs"].shape[0]
    for _ in range(args.num_chain):
        u = sample_from_prior(rng, data)
        v = rng.standard_normal(dim_y)
        if args.algorithm == "chmc":
            params, x = generate_from_model(u, v, data)
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
    common.add_ssm_specific_args(parser)
    args = parser.parse_args()

    # Load data

    data = dict(
        onp.load(os.path.join(args.data_dir, "latent-garch-simulated-data.npz"))
    )
    data["y_obs"] = data["x_obs"] + args.obs_noise_std * data["n_obs"]
    dim_u = compute_dim_u(data)
    dim_y = data["y_obs"].shape[0]

    # Set up seeded random number generator

    rng = onp.random.default_rng(args.seed)

    # Define variables to be traced

    trace_func = common.construct_trace_func(generate_params, data, dim_u, dim_v=dim_y)

    # Run experiment

    (
        constrained_system_class,
        constrained_system_kwargs,
    ) = common.get_ssm_constrained_system_class_and_kwargs(
        args.use_manual_constraint_and_jacobian,
        generate_params,
        generate_x_0,
        forward_func,
        inverse_observation_func,
        constr_split,
        jacob_constr_split_blocks,
    )
    constrained_system_kwargs.update(data=data, dim_u=dim_u)

    final_states, traces, stats, summary_dict, sampler = common.run_experiment(
        args=args,
        data=data,
        dim_u=dim_u,
        rng=rng,
        experiment_name="latent_garch",
        dir_prefix=f"σ_{args.obs_noise_std:.0e}",
        var_names=list(prior_specifications.keys()),
        var_trace_func=trace_func,
        posterior_neg_log_dens=posterior_neg_log_dens,
        extended_prior_neg_log_dens=extended_prior_neg_log_dens,
        constrained_system_class=constrained_system_class,
        constrained_system_kwargs=constrained_system_kwargs,
        sample_initial_states=sample_initial_states,
    )

