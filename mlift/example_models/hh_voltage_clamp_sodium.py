"""Hodgkin-Huxley conductance-based neuronal model under a voltage clamp.

Sodium-channel based conductances.
"""

import os
import pickle
import numpy as onp
import jax.config
import jax.numpy as np
import jax.lax as lax
import jax.api as api
from mlift.systems import IndependentAdditiveNoiseModelSystem
from mlift.distributions import log_normal
from mlift.prior import PriorSpecification, set_up_prior
import mlift.example_models.utils as utils

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


prior_specifications = {
    "k_alpha_m_1": PriorSpecification(distribution=log_normal(-3, 1)),
    "k_alpha_m_2": PriorSpecification(distribution=log_normal(2, 1)),
    "k_alpha_m_3": PriorSpecification(distribution=log_normal(2, 1)),
    "k_beta_m_1": PriorSpecification(distribution=log_normal(0, 1)),
    "k_beta_m_2": PriorSpecification(distribution=log_normal(2, 1)),
    "k_alpha_h_1": PriorSpecification(distribution=log_normal(-3, 1)),
    "k_alpha_h_2": PriorSpecification(distribution=log_normal(2, 1)),
    "k_beta_h_1": PriorSpecification(distribution=log_normal(2, 1)),
    "k_beta_h_2": PriorSpecification(distribution=log_normal(2, 1)),
    "g_bar_Na": PriorSpecification(distribution=log_normal(2, 1)),
    "σ": PriorSpecification(distribution=log_normal(0, 1)),
}

compute_dim_u, generate_params, prior_neg_log_dens, sample_from_prior = set_up_prior(
    prior_specifications
)


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
    conductances = [
        solve_for_sodium_conductances(t_seq, -v, params)
        for t_seq, v in zip(data["obs_times_g_Na"], data["depolarizations"])
    ]
    x = np.concatenate(conductances)
    return params, x


def generate_y(u, n, data):
    params, x = generate_from_model(u, data)
    return x + params["σ"] * n


def extended_prior_neg_log_dens(q, data):
    dim_u = compute_dim_u(data)
    u, n = q[:dim_u], q[dim_u:]
    return prior_neg_log_dens(u, data) + (n ** 2).sum() / 2


def posterior_neg_log_dens(u, data):
    params, x = generate_from_model(u, data)
    return prior_neg_log_dens(u, data) + (
        np.sum(((x - data["y_obs"]) / params["σ"]) ** 2) / 2
        + x.shape[0] * np.log(params["σ"])
    )


def sample_initial_states(rng, args, data):
    """Sample initial states from prior."""
    init_states = []
    for _ in range(args.num_chain):
        u = sample_from_prior(rng, data)
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

    parser = utils.set_up_argparser_with_standard_arguments(
        "Run Hodgkin-Huxley model voltage-clamp data experiment"
    )
    args = parser.parse_args()

    # Load data

    with open(os.path.join(args.data_dir, "hodgkin-huxley-data.pkl"), "r+b") as f:
        data = pickle.load(f)

    data["y_obs"] = np.concatenate(data["obs_vals_g_Na"])
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
        experiment_name="hh_voltage_clamp_sodium",
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

