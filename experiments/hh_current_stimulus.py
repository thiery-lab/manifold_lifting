"""Conductance-based neuronal model subject to current stimulus."""

import os
import numpy as onp
import jax.config
import jax.numpy as np
import jax.lax as lax
import jax.api as api
import mlift
from mlift.distributions import normal, log_normal
from experiments import common

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


prior_specifications = {
    "k_tau_p_1": common.PriorSpecification(distribution=log_normal(8, 1)),
    "g_bar_Na": common.PriorSpecification(distribution=log_normal(4, 1)),
    "g_bar_K": common.PriorSpecification(distribution=log_normal(2, 1)),
    "g_bar_M": common.PriorSpecification(distribution=log_normal(-3, 1)),
    "g_leak": common.PriorSpecification(distribution=log_normal(-3, 1)),
    "v_t": common.PriorSpecification(distribution=normal(-60, 10)),
    "σ": common.PriorSpecification(distribution=log_normal(0, 1)),
}

(
    compute_dim_u,
    generate_params,
    prior_neg_log_dens,
    sample_from_prior,
) = common.set_up_prior(prior_specifications)


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


def calculate_spike_times(y, data, smoothing_window=10):
    v = onp.convolve(y, onp.ones(smoothing_window) / smoothing_window, "same")
    v = onp.clip(v, -10, None)
    v[onp.where(onp.diff(v) < 0)] = -10
    spike_times = data["t_obs"][onp.where(onp.diff(v) < 0)]
    return spike_times


def sample_initial_states(rng, args, data):
    """Sample initial state using approximate Bayesian computation reject type approach.

    Use an approximate Bayesian computation type approach of repeatedly sampling from
    prior until peak times of sequence (without noise) generated from state matches peak
    times of (noisy) observation sequence to within a tolerance. This helps to avoid
    chains getting trapped in 'bad' modes.
    """
    init_states = []
    peak_times_obs = common.calculate_peak_times(data["y_obs"], data["t_obs"], 10, -10)
    num_tries = 0
    jitted_generate_from_model = api.jit(api.partial(generate_from_model, data=data))
    while len(init_states) < args.num_chain and num_tries < args.max_init_tries:
        u = sample_from_prior(rng, data)
        params, x = jitted_generate_from_model(u)
        if not onp.all(onp.isfinite(x)):
            num_tries += 1
            continue
        peak_times = common.calculate_peak_times(x, data["t_obs"], 10, -10)
        n = (data["y_obs"] - x) / params["σ"]
        if not (
            peak_times.shape[0] == peak_times_obs.shape[0]
            and abs(peak_times - peak_times_obs).max()
            < args.init_peak_time_diff_threshold
            and (n**2).mean() < 100
        ):
            num_tries += 1
            continue
        if args.algorithm == "chmc":
            q = onp.concatenate((u, onp.asarray(n)))
            assert (
                abs(x + params["σ"] * n - data["y_obs"]).max()
                < args.projection_solver_warm_up_constraint_tol
            )
        else:
            q = u
        init_states.append(q)
        num_tries += 1
    if len(init_states) != args.num_chain:
        raise RuntimeError(
            f"Failed to find {args.num_chain} acceptable initial states in "
            f"{args.max_init_tries} tries."
        )
    return init_states


if __name__ == "__main__":

    # Process command line arguments defining experiment parameters

    parser = common.set_up_argparser_with_standard_arguments(
        "Run Hodgkin-Huxley model current stimulus simulated data experiment"
    )
    parser.add_argument(
        "--obs-noise-std",
        type=float,
        default=1.0,
        help="Standard deviation of observation noise to use in simulated data",
    )
    parser.add_argument(
        "--max-init-tries",
        type=int,
        default=10000,
        help="Maximum number of prior samples to try to find acceptable initial states",
    )
    parser.add_argument(
        "--init-peak-time-diff-threshold",
        type=float,
        default=2.0,
        help="Maximum difference between peak times of initial state and observations",
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

    trace_func = common.construct_trace_func(generate_params, data, dim_u)

    # Run experiment

    final_states, traces, stats, summary_dict, sampler = common.run_experiment(
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
        constrained_system_class=mlift.IndependentAdditiveNoiseModelSystem,
        constrained_system_kwargs={
            "generate_y": generate_y,
            "data": data,
            "dim_u": dim_u,
        },
        sample_initial_states=sample_initial_states,
    )

