import os
import logging
import argparse
import datetime
import json
import pickle
import time
import mici
import mlift
from mlift.transforms import normal_to_half_cauchy, normal_to_uniform
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
    description="Run eight-school hierarchical model experiment"
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
args = parser.parse_args()

# Set up output directory

timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
algorithm_subtype = (
    f"{args.metric_type}_metric"
    if args.algorithm == "hmc"
    else f"{args.projection_solver}_solver"
)
dir_name = f"{args.algorithm}_{algorithm_subtype}_{timestamp}"
output_dir = os.path.join(args.output_root_dir, "eight_schools", dir_name)

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

data = dict(np.load(os.path.join(args.data_dir, "eight-schools-data.npz")))


# Define model functions

dim_u = 2
dim_y = data["y_obs"].shape[0]


def generate_params(u):
    return {
        "μ": u[0] * 5,
        "τ": normal_to_half_cauchy(u[1]) * 5,
    }


def generate_x(u, v):
    params = generate_params(u)
    x = params["μ"] + params["τ"] * v
    return params, x


def generate_from_model(u, v, n, data):
    params, x = generate_x(u, v)
    y = x + data["σ"] * n
    return params, x, y


def generate_y(u, v, n, data):
    _, _, y = generate_from_model(u, v, n, data)
    return y


def _neg_log_dens_hmc(q):
    u, v = q[:dim_u], q[dim_u:]
    _, x = generate_x(u, v)
    return (((data["y_obs"] - x) / data["σ"]) ** 2 / 2 + np.log(data["σ"])).sum() + (
        q ** 2
    ).sum() / 2


_grad_neg_log_dens_hmc = api.jit(api.value_and_grad(_neg_log_dens_hmc))


def neg_log_dens_hmc(q):
    return onp.asarray(_neg_log_dens_hmc(q))


def grad_neg_log_dens_hmc(q):
    val, grad = _grad_neg_log_dens_hmc(q)
    return onp.asarray(grad), onp.asarray(val)


# Set up Mici objects

if args.algorithm == "chmc":

    system = mlift.HierarchicalLatentVariableModelSystem(
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
    u, v = state.pos[:dim_u], state.pos[dim_u:dim_u+dim_y]
    params, x = generate_x(u, v)
    h = system.h(state)
    call_counts = {
        name.split(".")[-1] + "_calls": val
        for (name, _), val in state._call_counts.items()
    }
    return {**params, **call_counts, "x": x, "hamiltonian": h, "u": u}


# Initialise chain states

init_states = []
for c in range(args.num_chain):
    u = rng.standard_normal(dim_u)
    v = rng.standard_normal(dim_y)
    if args.algorithm == "chmc":
        params, x = generate_x(u, v)
        n = (data["y_obs"] - x) / data["σ"]
        q = onp.concatenate((u, v, onp.asarray(n)))
        assert (
            abs(x + data["σ"] * n - data["y_obs"]).max()
            < args.projection_solver_warm_up_constraint_tol
        )
    else:
        q = onp.concatenate((u, v))
    init_states.append(mici.states.ChainState(pos=q, mom=None, dir=1, _call_counts={}))


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
summary_vars = ["μ", "τ", "x", "hamiltonian"]
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
