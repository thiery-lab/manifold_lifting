import os
import logging
import argparse
import datetime
import json
import pickle
import time
import mici
import mlift
import numpy as onp
import jax.config
import jax.numpy as np
import jax.lax as lax
import jax.api as api
import arviz

# Ensure Jax configured to use double-precision and to run on CPU

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# Process command line arguments defining experiment parameters

parser = argparse.ArgumentParser(
    description="Run Hodgkin-Huxley model current stimulus simulated data experiment"
)
parser.add_argument(
    "--output-root-dir",
    default="results",
    help="Root directory to make experiment output subdirectory in",
)
parser.add_argument(
    "--data-dir", default="data", help="Directory containing dataset files",
)
parser.add_argument(
    "--obs-noise-std",
    type=float,
    default=2.0,
    help="Standard deviation of observation noise to use in simulated data",
)
parser.add_argument(
    "--algorithm",
    default="chmc",
    choices=("chmc", "hmc"),
    help="Which algorithm to perform inference with, from: chmc, hmc",
)
parser.add_argument(
    "--seed", type=int, default=202101, help="Seed for random number generator"
)
parser.add_argument(
    "--num-chain", type=int, default=4, help="Number of independent chains to sample"
)
parser.add_argument(
    "--num-warm-up-iter",
    type=int,
    default=1000,
    help="Number of chain iterations in adaptive warm-up sampling stage",
)
parser.add_argument(
    "--num-main-iter",
    type=int,
    default=2500,
    help="Number of chain iterations in main sampling stage",
)
parser.add_argument(
    "--step-size-adaptation-target",
    type=float,
    default=0.8,
    help="Target acceptance statistic for step size adaptation",
)
parser.add_argument(
    "--step-size-reg-coefficient",
    type=float,
    default=0.1,
    help="Regularisation coefficient for step size adaptation",
)
parser.add_argument(
    "--metric-type",
    choices=("diagonal", "dense"),
    default="diagonal",
    help=(
        "Metric type to adaptively tune during warm-up stage when using HMC algorithm. "
        "If 'diagonal' a diagonal metric matrix representation is used with diagonal "
        "entries set to reciprocals of estimates of the marginal posterior variances. "
        "If 'dense' a dense metric matrix representation is used corresponding to the "
        "inverse of an estimate of the posterior covariance matrix."
    ),
)
parser.add_argument(
    "--projection-solver",
    choices=("newton", "quasi-newton"),
    default="newton",
    help="Iterative method to solve projection onto manifold when using CHMC algorithm.",
)
parser.add_argument(
    "--projection-solver-max-iters",
    type=int,
    default=50,
    help="Maximum number of iterations to try in projection solver",
)
parser.add_argument(
    "--projection-solver-warm-up-constraint-tol",
    type=float,
    default=1e-6,
    help="Warm-up stage tolerance for constraint function norm in projection solver",
)
parser.add_argument(
    "--projection-solver-warm-up-position-tol",
    type=float,
    default=1e-5,
    help="Warm-up stage tolerance for change in position norm in projection solver",
)
parser.add_argument(
    "--projection-solver-main-constraint-tol",
    type=float,
    default=1e-9,
    help="Main stage tolerance for constraint function norm in projection solver",
)
parser.add_argument(
    "--projection-solver-main-position-tol",
    type=float,
    default=1e-8,
    help="Main stage tolerance for change in position norm in projection solver",
)
parser.add_argument(
    "--max-init-tries",
    type=int,
    default=10000,
    help="Maximum number of prior samples to try to find acceptable initial states",
)
parser.add_argument(
    "--init-spike-time-diff-threshold",
    type=int,
    default=1,
    help="Maximum difference between spike times of initial state and observations",
)
args = parser.parse_args()

# Set up output directory

timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
algorithm_subtype = (
    f"{args.metric_type}_metric"
    if args.algorithm == "hmc"
    else f"{args.projection_solver}_solver"
)
dir_name = (
    f"σ_{args.obs_noise_std:.0e}_{args.algorithm}_{algorithm_subtype}_{timestamp}"
)
output_dir = os.path.join(args.output_root_dir, "hh_current_stimulus", dir_name)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(os.path.join(output_dir, "args.json"), "w") as f:
    json.dump(vars(args), f, indent=2)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = []
logger.addHandler(logging.FileHandler(os.path.join(output_dir, "info.log")))


# Set up seeded random number generator

rng = onp.random.default_rng(args.seed)

# Load data

data = dict(
    onp.load(os.path.join(args.data_dir, "hodgkin-huxley-current-stimulus-data.npz"))
)
data["y_obs"] = data["x_obs"] + args.obs_noise_std * data["n_obs"]


# Define model functions


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


params_prior_mean_std_exp_transform = {
    "k_tau_p_1": (8.0, 1.0, True),
    "g_bar_Na": (4.0, 1.0, True),
    "g_bar_K": (2.0, 1.0, True),
    "g_bar_M": (-3.0, 1.0, True),
    "g_leak": (-3.0, 1.0, True),
    "v_t": (-60.0, 10.0, False),
    "σ": (1.0, 1.0, True),
}

dim_u = len(params_prior_mean_std_exp_transform)


def generate_params(u, data):
    params = {}
    for i, (name, (mean, std, exp_transform)) in enumerate(
        params_prior_mean_std_exp_transform.items()
    ):
        if exp_transform:
            params[name] = np.exp(mean + u[i] * std)
        else:
            params[name] = mean + u[i] * std
    return params


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


@api.jit
def _neg_log_dens_hmc(u):
    params, y = generate_from_model(u, data)
    return (
        np.sum(((y - data["y_obs"]) / params["σ"]) ** 2) / 2
        + y.shape[0] * np.log(params["σ"])
        + np.sum(u ** 2) / 2
    )


_val_and_grad_neg_log_dens_hmc = api.jit(api.value_and_grad(_neg_log_dens_hmc))


def neg_log_dens_hmc(q):
    return onp.asarray(_neg_log_dens_hmc(q))


def grad_neg_log_dens_hmc(q):
    val, grad = _val_and_grad_neg_log_dens_hmc(q)
    return onp.asarray(grad), onp.asarray(val)


# Set up Mici objects

if args.algorithm == "chmc":

    system = mlift.IndependentAdditiveNoiseModelSystem(
        generate_y=generate_y, data=data, dim_u=dim_u
    )

    projection_solver = (
        mlift.jitted_solve_projection_onto_manifold_newton
        if args.projection_solver == "newton"
        else mlift.jitted_solve_projection_onto_manifold_quasi_newton
    )

    projection_solver_kwargs = {
        "convergence_tol": args.projection_solver_warm_up_constraint_tol,
        "position_tol": args.projection_solver_warm_up_position_tol,
        "max_iters": args.projection_solver_max_iters,
    }

    integrator = mici.integrators.ConstrainedLeapfrogIntegrator(
        system,
        projection_solver=projection_solver,
        reverse_check_tol=2 * args.projection_solver_warm_up_position_tol,
        projection_solver_kwargs=projection_solver_kwargs,
    )

    tolerance_adapter = mlift.ToleranceAdapter(
        warm_up_constraint_tol=args.projection_solver_warm_up_constraint_tol,
        warm_up_position_tol=args.projection_solver_warm_up_position_tol,
        main_constraint_tol=args.projection_solver_main_constraint_tol,
        main_position_tol=args.projection_solver_main_position_tol,
    )

    adapters = [tolerance_adapter]

    monitor_stats = [
        "accept_stat",
        "convergence_error",
        "non_reversible_step",
        "n_step",
    ]

else:

    system = mici.systems.EuclideanMetricSystem(
        neg_log_dens=neg_log_dens_hmc, grad_neg_log_dens=grad_neg_log_dens_hmc
    )

    integrator = mici.integrators.LeapfrogIntegrator(system)

    metric_adapter = (
        mici.adapters.OnlineVarianceMetricAdapter()
        if args.metric_type == "diagonal"
        else mici.adapters.OnlineCovarianceMetricAdapter()
    )

    adapters = [metric_adapter]

    monitor_stats = ["accept_stat", "diverging", "n_step"]

step_size_adapter = mici.adapters.DualAveragingStepSizeAdapter(
    adapt_stat_target=args.step_size_adaptation_target,
    log_step_size_reg_coefficient=args.step_size_reg_coefficient,
)

adapters.append(step_size_adapter)

sampler = mici.samplers.DynamicMultinomialHMC(system, integrator, rng)


def trace_func(state):
    u = state.pos[:dim_u]
    params = generate_params(u, data)
    h = system.h(state)
    call_counts = {
        name.split(".")[-1] + "_calls": val
        for (name, _), val in state._call_counts.items()
    }
    return {**params, **call_counts, "hamiltonian": h, "u": u}


# Initialise chain states
# Use a heuristic of repeatedly sampling from prior until number of spikes in generated
# sequence (wihout noise) matches (noisy) observation sequence. This helps to avoid
# getting trapped in 'bad' modes

print("Finding acceptable initial states ...")


def calculate_spike_times(y, data, smoothing_window=10):
    v = onp.convolve(y, onp.ones(smoothing_window) / smoothing_window, "same")
    v = onp.clip(v, -10, None)
    v[onp.where(onp.diff(v) < 0)] = -10
    spike_times = data["t_obs"][onp.where(onp.diff(v) < 0)]
    return spike_times


init_states = []
spike_times_obs = calculate_spike_times(data["y_obs"], data)
num_tries = 0
jitted_generate_from_model = api.jit(lambda u: generate_from_model(u, data))
while len(init_states) < args.num_chain and num_tries < args.max_init_tries:
    u = rng.standard_normal(dim_u)
    params, x = jitted_generate_from_model(u)
    spike_times = calculate_spike_times(x, data)
    if (
        spike_times.shape[0] == spike_times_obs.shape[0]
        and abs(spike_times - spike_times_obs).max()
        < args.init_spike_time_diff_threshold
    ):
        if args.algorithm == "chmc":
            n = (data["y_obs"] - x) / params["σ"]
            q = onp.concatenate((u, onp.asarray(n)))
            assert (
                abs(x + params["σ"] * n - data["y_obs"]).max()
                < args.projection_solver_warm_up_constraint_tol
            )
        else:
            q = u
        init_states.append(
            mici.states.ChainState(pos=q, mom=None, dir=1, _call_counts={})
        )
    num_tries += 1

if len(init_states) != args.num_chain:
    raise RuntimeError(
        f"Failed to find {args.num_chain} acceptable initial states in "
        f"{args.max_init_tries} tries."
    )


# Precompile JAX functions to avoid compilation time appearing in chain run times

print("Pre-compiling JAX functions ...")

start_time = time.time()

if args.algorithm == "chmc":
    system.precompile_jax_functions(q)
else:
    neg_log_dens_hmc(q)
    grad_neg_log_dens_hmc

compile_time = time.time() - start_time
print(f"Total compile time = {compile_time:.0f} seconds")

# Sample chains

# Ignore NumPy floating point overflow warnings
# Prevents warning messages being produced while progress bars are being printed

onp.seterr(over="ignore")

start_time = time.time()

final_states, traces, stats = sampler.sample_chains_with_adaptive_warm_up(
    args.num_warm_up_iter,
    args.num_main_iter,
    init_states,
    trace_funcs=[trace_func],
    adapters=adapters,
    memmap_enabled=True,
    memmap_path=output_dir,
    monitor_stats=monitor_stats,
    max_threads_per_process=1,
)

sampling_time = time.time() - start_time
summary_vars = list(params_prior_mean_std_exp_transform.keys()) + ["hamiltonian"]
summary = arviz.summary(traces, var_names=summary_vars)
summary_dict = summary.to_dict()
summary_dict["total_compile_time"] = compile_time
summary_dict["total_sampling_time"] = sampling_time
summary_dict["final_integrator_step_size"] = integrator.step_size
for key, value in traces.items():
    if key[-6:] == "_calls":
        summary_dict["total_" + key] = sum(int(v[-1]) for v in value)
with open(os.path.join(output_dir, "summary.json"), mode="w") as f:
    json.dump(summary_dict, f, ensure_ascii=False, indent=2)

print(f"Integrator step size = {integrator.step_size:.2g}")
print(f"Total sampling time = {sampling_time:.0f} seconds")
print(summary)
