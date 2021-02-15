"""Hodgkin-Huxley conductance-based neuronal model under a voltage clamp."""

import os
import pickle
import numpy as onp
import jax.config
import jax.numpy as np
import jax.lax as lax
import jax.api as api
import mlift
from experiments import common

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


def alpha_n(v, params):
    return (
        params["k_alpha_n_1"]
        * (v + params["k_alpha_n_2"])
        / (np.exp((v + params["k_alpha_n_2"]) / params["k_alpha_n_3"]) - 1)
    )


def beta_n(v, params):
    return params["k_beta_n_1"] * np.exp(v / params["k_beta_n_2"])


def alpha_m(v, params):
    return (
        params["k_alpha_m_1"]
        * (v + params["k_alpha_m_2"])
        / (np.exp((v + params["k_alpha_m_2"]) / params["k_alpha_m_3"]) - 1)
    )


def beta_m(v, params):
    return params["k_beta_m_1"] * np.exp(v / params["k_beta_m_2"])


def alpha_h(v, params):
    return params["k_alpha_h_1"] * np.exp(v / params["k_alpha_h_2"])


def beta_h(v, params):
    return 1 / (np.exp((v + params["k_beta_h_1"]) / params["k_beta_h_2"]) + 1)


param_sets = {
    "potassium": {
        "k_alpha_n_1",
        "k_alpha_n_2",
        "k_alpha_n_3",
        "k_beta_n_1",
        "k_beta_n_2",
        "g_bar_K",
        "σ",
    },
    "sodium": {
        "k_alpha_m_1",
        "k_alpha_m_2",
        "k_alpha_m_3",
        "k_beta_m_1",
        "k_beta_m_2",
        "k_alpha_h_1",
        "k_alpha_h_2",
        "k_beta_h_1",
        "k_beta_h_2",
        "g_bar_Na",
        "σ",
    },
}

param_sets["both"] = param_sets["potassium"] | param_sets["sodium"]

log_param_prior_mean_stds = {
    "k_alpha_n_1": (-3, 1),
    "k_alpha_n_2": (2, 1),
    "k_alpha_n_3": (2, 1),
    "k_beta_n_1": (-3, 1),
    "k_beta_n_2": (2, 1),
    "k_alpha_m_1": (-3, 1),
    "k_alpha_m_2": (2, 1),
    "k_alpha_m_3": (2, 1),
    "k_beta_m_1": (0, 1),
    "k_beta_m_2": (2, 1),
    "k_alpha_h_1": (-3, 1),
    "k_alpha_h_2": (2, 1),
    "k_beta_h_1": (2, 1),
    "k_beta_h_2": (2, 1),
    "g_bar_K": (2, 1),
    "g_bar_Na": (2, 1),
    "σ": (0, 1),
}


def generate_params(u, data):
    params = {}
    for i, param_name in enumerate(param_sets[data["dataset"]]):
        m, s = log_param_prior_mean_stds[param_name]
        params[param_name] = np.exp(m + u[i] * s)
    return params


def solve_for_potassium_conductances(t_seq, v, params):
    n_0 = alpha_n(0, params) / (alpha_n(0, params) + beta_n(0, params))
    a_n, b_n = alpha_n(v, params), beta_n(v, params)
    n_infty, tau_n = a_n / (a_n + b_n), 1 / (a_n + b_n)
    n_seq = n_0 - (n_0 - n_infty) * (1 - np.exp(-t_seq / tau_n))
    return params["g_bar_K"] * n_seq ** 4


def solve_for_sodium_conductances(t_seq, v, params):
    m_0 = alpha_m(0, params) / (alpha_m(0, params) + beta_m(0, params))
    h_0 = alpha_h(0, params) / (alpha_h(0, params) + beta_h(0, params))
    a_m, b_m = alpha_m(v, params), beta_m(v, params)
    m_infty, tau_m = a_m / (a_m + b_m), 1 / (a_m + b_m)
    m_seq = m_0 - (m_0 - m_infty) * (1 - np.exp(-t_seq / tau_m))
    a_h, b_h = alpha_h(v, params), beta_h(v, params)
    h_infty, tau_h = a_h / (a_h + b_h), 1 / (a_h + b_h)
    h_seq = h_0 - (h_0 - h_infty) * (1 - np.exp(-t_seq / tau_h))
    return params["g_bar_Na"] * m_seq ** 3 * h_seq


def generate_from_model(u, data):
    params = generate_params(u, data)
    conductances = []
    if data["dataset"] in ["potassium", "both"]:
        conductances += [
            solve_for_potassium_conductances(t_seq, -v, params)
            for t_seq, v in zip(data["obs_times_g_K"], data["depolarizations"])
        ]
    if data["dataset"] in ["sodium", "both"]:
        conductances += [
            solve_for_sodium_conductances(t_seq, -v, params)
            for t_seq, v in zip(data["obs_times_g_Na"], data["depolarizations"])
        ]
    x = np.concatenate(conductances)
    return params, x


def generate_y(u, n, data):
    params, x = generate_from_model(u, data)
    return x + params["σ"] * n


def posterior_neg_log_dens(u, data):
    params, x = generate_from_model(u, data)
    return (
        np.sum(((x - data["y_obs"]) / params["σ"]) ** 2) / 2
        + x.shape[0] * np.log(params["σ"])
        + np.sum(u ** 2) / 2
    )


def sample_initial_states(rng, args, data):
    """Sample initial states from prior."""
    init_states = []
    for c in range(args.num_chain):
        u = rng.standard_normal(data["dim_u"])
        if args.algorithm == "chmc":
            params, x = generate_from_model(u, data)
            n = (data["y_obs"] - x) / params["σ"]
            q = onp.concatenate((u, onp.asarray(n)))
            assert (
                abs(x + params["σ"] * n - data["y_obs"]).max()
                < args.projection_solver_warm_up_constraint_tol
            )
        else:
            q = u
        init_states.append(q)
    return init_states


if __name__ == "__main__":

    # Process command line arguments defining experiment parameters

    parser = common.set_up_argparser_with_standard_arguments(
        "Run Hodgkin-Huxley model voltage-clamp data experiment"
    )
    parser.add_argument(
        "--dataset",
        default="sodium",
        choices=("potassium", "sodium", "both"),
        help="Which conductance data to perform inference with",
    )
    args = parser.parse_args()

    # Load data

    with open(os.path.join(args.data_dir, "hodgkin-huxley-data.pkl"), "r+b") as f:
        data = pickle.load(f)

    if args.dataset == "potassium":
        data["y_obs"] = np.concatenate(data["obs_vals_g_K"])
    elif args.dataset == "sodium":
        data["y_obs"] = np.concatenate(data["obs_vals_g_Na"])
    elif args.dataset == "both":
        data["y_obs"] = np.concatenate(data["obs_vals_g_K"] + data["obs_vals_g_Na"])
    else:
        raise ValueError(f"Unrecognised dataset: {args.dataset}")
    var_names = list(param_sets[args.dataset])
    dim_u = len(var_names)
    data["dim_u"] = dim_u
    data["dataset"] = args.dataset

    # Set up seeded random number generator

    rng = onp.random.default_rng(args.seed)

    # Define variables to be traced

    def trace_func(state):
        u = state.pos[:dim_u]
        params = generate_params(u, data)
        return {**params, "u": u}

    # Run experiment

    final_states, traces, stats, summary_dict, sampler = common.run_experiment(
        args=args,
        data=data,
        dim_u=dim_u,
        rng=rng,
        experiment_name="hh_voltage_clamp",
        dir_prefix=f"{args.dataset}_data",
        var_names=var_names,
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

