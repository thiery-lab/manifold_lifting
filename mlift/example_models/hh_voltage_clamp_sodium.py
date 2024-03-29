"""Hodgkin-Huxley conductance-based neuronal model under a voltage clamp.

Sodium-channel based conductances.
"""

import os
import numpy as onp
import jax
import jax.config
import jax.numpy as np
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
    "k_beta_h_2": PriorSpecification(distribution=log_normal(1, 1)),
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


def solve_for_sodium_conductance(t, v, params):
    m_0 = alpha_m(0, params) / (alpha_m(0, params) + beta_m(0, params))
    h_0 = alpha_h(0, params) / (alpha_h(0, params) + beta_h(0, params))
    a_m, b_m = alpha_m(v, params), beta_m(v, params)
    m_infty, tau_m = a_m / (a_m + b_m), 1 / (a_m + b_m)
    m = m_0 - (m_0 - m_infty) * (1 - np.exp(-t / tau_m))
    a_h, b_h = alpha_h(v, params), beta_h(v, params)
    h_infty, tau_h = a_h / (a_h + b_h), 1 / (a_h + b_h)
    h = h_0 - (h_0 - h_infty) * (1 - np.exp(-t / tau_h))
    return params["g_bar_Na"] * m ** 3 * h


def generate_from_model(u, data):
    params = generate_params(u, data)
    x = jax.vmap(solve_for_sodium_conductance, (0, 0, None))(
        data["times"], -data["depolarizations"], params
    )
    return params, x


def generate_y(u, n, data):
    params, x = generate_from_model(u, data)
    return x + params["σ"] * n


def jacob_generate_y(u, n, data):

    def g_y(u, n, t, v):
        params = generate_params(u, data)
        return solve_for_sodium_conductance(t, v, params) + params["σ"] * n

    y, (dy_du, dy_dn) = jax.vmap(jax.value_and_grad(g_y, (0, 1)), (None, 0, 0, 0))(
        u, n, data["times"], -data["depolarizations"]
    )
    return (dy_du, dy_dn), y


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


def sample_initial_states(rng, data, num_chain=4, algorithm="chmc"):
    """Sample initial states from prior."""
    init_states = []
    for _ in range(num_chain):
        u = sample_from_prior(rng, data)
        if algorithm == "chmc":
            params, x = generate_from_model(u, data)
            n = (data["y_obs"] - x) / params["σ"]
            q = onp.concatenate((u, onp.asarray(n)))
        else:
            q = onp.asarray(u)
        init_states.append(q)
    return init_states


if __name__ == "__main__":

    # Process command line arguments defining experiment parameters

    parser = utils.set_up_argparser_with_standard_arguments(
        "Run Hodgkin-Huxley model voltage-clamp data experiment"
    )
    parser.add_argument(
        "--use-manual-jacobian",
        action="store_true",
        help="Use manually constructed generator function Jacobian",
    )
    args = parser.parse_args()

    # Load data

    data = dict(onp.load(os.path.join(args.data_dir, "hodgkin-huxley-sodium-data.npz")))
    data["y_obs"] = data["conductances"]
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
            "jacob_generate_y": jacob_generate_y, # if args.use_manual_jacobian else None,
        },
        sample_initial_states=sample_initial_states,
    )

