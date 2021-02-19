"""Conductance-based neuronal model subject to current stimulus."""

import os
import numpy as onp
import jax.config
import jax.numpy as np
import jax.lax as lax
import jax.api as api
from mlift.systems import IndependentAdditiveNoiseModelSystem
from mlift.distributions import normal, log_normal
from mlift.prior import PriorSpecification, set_up_prior
import mlift.example_models.utils as utils

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


prior_specifications = {
    "k_tau_p_1": PriorSpecification(distribution=log_normal(8, 1)),
    "g_bar_Na": PriorSpecification(distribution=log_normal(4, 1)),
    "g_bar_K": PriorSpecification(distribution=log_normal(2, 1)),
    "g_bar_M": PriorSpecification(distribution=log_normal(-3, 1)),
    "g_leak": PriorSpecification(distribution=log_normal(-3, 1)),
    "v_t": PriorSpecification(distribution=normal(-60, 10)),
    "σ": PriorSpecification(distribution=log_normal(0, 1)),
}

compute_dim_u, generate_params, prior_neg_log_dens, sample_from_prior = set_up_prior(
    prior_specifications
)


def x_over_expm1_x(x):
    return x / np.expm1(x)


def alpha_n(v, params_and_data):
    return (
        params_and_data["k_alpha_n_1"]
        * x_over_expm1_x(
            -(v - params_and_data["v_t"] - params_and_data["k_alpha_n_2"])
            / params_and_data["k_alpha_n_3"]
        )
        * params_and_data["k_alpha_n_3"]
    )


def beta_n(v, params_and_data):
    return params_and_data["k_beta_n_1"] * np.exp(
        -(v - params_and_data["v_t"] - params_and_data["k_beta_n_2"])
        / params_and_data["k_beta_n_3"]
    )


def alpha_m(v, params_and_data):
    return (
        params_and_data["k_alpha_m_1"]
        * x_over_expm1_x(
            -(v - params_and_data["v_t"] - params_and_data["k_alpha_m_2"])
            / params_and_data["k_alpha_m_3"]
        )
        * params_and_data["k_alpha_m_3"]
    )


def beta_m(v, params_and_data):
    return (
        params_and_data["k_beta_m_1"]
        * x_over_expm1_x(
            (v - params_and_data["v_t"] - params_and_data["k_beta_m_2"])
            / params_and_data["k_beta_m_3"]
        )
        * params_and_data["k_beta_m_3"]
    )


def alpha_h(v, params_and_data):
    return params_and_data["k_alpha_h_1"] * np.exp(
        -(v - params_and_data["v_t"] - params_and_data["k_alpha_h_2"])
        / params_and_data["k_alpha_h_3"]
    )


def beta_h(v, params_and_data):
    return params_and_data["k_beta_h_1"] / (
        np.exp(
            -(v - params_and_data["v_t"] - params_and_data["k_beta_h_2"])
            / params_and_data["k_beta_h_3"]
        )
        + 1
    )


def p_infinity(v, params_and_data):
    return 1 / (
        1
        + np.exp(
            (-v + params_and_data["k_p_infinity_1"]) / params_and_data["k_p_infinity_2"]
        )
    )


def tau_p(v, params_and_data):
    return params_and_data["k_tau_p_1"] / (
        params_and_data["k_tau_p_2"]
        * np.exp((v + params_and_data["k_tau_p_3"]) / params_and_data["k_tau_p_4"])
        + np.exp(-(v + params_and_data["k_tau_p_3"]) / params_and_data["k_tau_p_4"])
    )


def i_stimulus(t, params_and_data):
    return lax.cond(
        np.logical_or(
            np.less(t, params_and_data["stimulus_start_time"]),
            np.greater(t, params_and_data["stimulus_end_time"]),
        ),
        lambda _: 0.0,
        lambda _: params_and_data["i_stimulus"],
        None,
    )


def generate_x_init(params_and_data):
    # Initialise at steady-state values
    v = params_and_data["E_leak"]
    return np.array(
        [
            v,  # v
            alpha_n(v, params_and_data)
            / (alpha_n(v, params_and_data) + beta_n(v, params_and_data)),  # n
            alpha_m(v, params_and_data)
            / (alpha_m(v, params_and_data) + beta_m(v, params_and_data)),  # m
            alpha_h(v, params_and_data)
            / (alpha_h(v, params_and_data) + beta_h(v, params_and_data)),  # h
            p_infinity(v, params_and_data),  # p
        ]
    )


def integrate_ode(x_init, params_and_data):
    def step_func(x, t_dt):
        t, dt = t_dt
        v, n, m, h, p = x
        g_K = params_and_data["g_bar_K"] * n ** 4
        g_Na = params_and_data["g_bar_Na"] * m ** 3 * h
        g_M = params_and_data["g_bar_M"] * p
        tau_v = params_and_data["c_m"] / (g_K + g_Na + g_M + params_and_data["g_leak"])
        v_infinity = (
            (
                g_K * params_and_data["E_K"]
                + g_Na * params_and_data["E_Na"]
                + g_M * params_and_data["E_K"]
                + params_and_data["g_leak"] * params_and_data["E_leak"]
                + i_stimulus(t, params_and_data)
            )
            * tau_v
            / params_and_data["c_m"]
        )
        v_ = v_infinity + (v - v_infinity) * np.exp(-dt / tau_v)
        a_n, b_n = alpha_n(v_, params_and_data), beta_n(v_, params_and_data)
        a_m, b_m = alpha_m(v_, params_and_data), beta_m(v_, params_and_data)
        a_h, b_h = alpha_h(v_, params_and_data), beta_h(v_, params_and_data)
        tau_n, n_infinity = 1 / (a_n + b_n), a_n / (a_n + b_n)
        tau_m, m_infinity = 1 / (a_m + b_m), a_m / (a_m + b_m)
        tau_h, h_infinity = 1 / (a_h + b_h), a_h / (a_h + b_h)
        n_ = n_infinity + (n - n_infinity) * np.exp(-dt / tau_n)
        m_ = m_infinity + (m - m_infinity) * np.exp(-dt / tau_m)
        h_ = h_infinity + (h - h_infinity) * np.exp(-dt / tau_h)
        p_infinity_ = p_infinity(v_, params_and_data)
        p_ = p_infinity_ + (p - p_infinity_) * np.exp(-dt / tau_p(v_, params_and_data))
        x_ = np.array([v_, n_, m_, h_, p_])
        return (x_, x_)

    n_step = int(params_and_data["end_time"] / params_and_data["dt"]) + 1
    t_seq = np.linspace(0, params_and_data["end_time"], n_step)
    dt_seq = t_seq[1:] - t_seq[:-1]
    _, x_seq = lax.scan(step_func, x_init, (t_seq[:-1], dt_seq))
    return np.concatenate((x_init[None], x_seq))


def generate_from_model(u, data):
    params = generate_params(u, data)
    params_and_data = {**data, **params}
    x_init = generate_x_init(params_and_data)
    x_seq = integrate_ode(x_init, params_and_data)
    obs_indices = slice(
        int(data["stimulus_start_time"] / data["dt"]),
        int(data["stimulus_end_time"] / data["dt"]) + 1,
        int(data["observation_interval"] / data["dt"]),
    )
    return params, x_seq[obs_indices, 0]


def generate_y(u, n, data):
    params, x = generate_from_model(u, data)
    return x + params["σ"] * n


def extended_prior_neg_log_dens(q, data):
    dim_u = compute_dim_u(data)
    u, n = q[:dim_u], q[dim_u:]
    return prior_neg_log_dens(u, data) + (n ** 2).sum() / 2


def posterior_neg_log_dens(u, data):
    params, y = generate_from_model(u, data)
    return (
        np.sum(((y - data["y_obs"]) / params["σ"]) ** 2) / 2
        + y.shape[0] * np.log(params["σ"])
        + np.sum(u ** 2) / 2
    )


def sample_initial_states(
    rng,
    data,
    num_chain=4,
    algorithm="chmc",
    max_init_tries=10000,
    peak_time_diff_threshold=2,
    smoothing_window_width=10,
    peak_value_threshold=-10,
    mean_residual_sq_threshold=100
):
    """Sample initial state using approximate Bayesian computation reject type approach.

    Use an approximate Bayesian computation type approach of repeatedly sampling from
    prior until peak times of sequence (without noise) generated from state matches peak
    times of (noisy) observation sequence to within a tolerance. This helps to avoid
    chains getting trapped in 'bad' modes.
    """
    init_states = []
    peak_times_obs = utils.calculate_peak_times(
        data["y_obs"], data["t_obs"], smoothing_window_width, peak_value_threshold
    )
    num_tries = 0
    jitted_generate_from_model = api.jit(api.partial(generate_from_model, data=data))
    while len(init_states) < num_chain and num_tries < max_init_tries:
        u = sample_from_prior(rng, data)
        params, x = jitted_generate_from_model(u)
        if not onp.all(onp.isfinite(x)):
            num_tries += 1
            continue
        peak_times = utils.calculate_peak_times(
            x, data["t_obs"], smoothing_window_width, peak_value_threshold
        )
        n = (data["y_obs"] - x) / params["σ"]
        if not (
            peak_times.shape[0] == peak_times_obs.shape[0]
            and abs(peak_times - peak_times_obs).max() < peak_time_diff_threshold
            and (n ** 2).mean() < mean_residual_sq_threshold
        ):
            num_tries += 1
            continue
        if algorithm == "chmc":
            q = onp.concatenate((u, onp.asarray(n)))
        else:
            q = onp.asarray(u)
        init_states.append(q)
        num_tries += 1
    if len(init_states) != num_chain:
        raise RuntimeError(
            f"Failed to find {num_chain} acceptable initial states in "
            f"{max_init_tries} tries."
        )
    return init_states


if __name__ == "__main__":

    # Process command line arguments defining experiment parameters

    parser = utils.set_up_argparser_with_standard_arguments(
        "Run Hodgkin-Huxley model current stimulus simulated data experiment"
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
        onp.load(
            os.path.join(args.data_dir, "hodgkin-huxley-current-stimulus-data.npz")
        )
    )
    data["y_obs"] = data["x_obs"] + args.obs_noise_std * data["n_obs"]
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
        experiment_name="hh_current_stimulus",
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

