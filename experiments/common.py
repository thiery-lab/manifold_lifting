import os
import logging
import argparse
import datetime
import json
import pickle
import time
import timeit
import arviz
import numpy as np
import mici
import mlift


def set_up_argparser_with_standard_arguments(description):
    parser = argparse.ArgumentParser(description=description)
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
        "--num-chain",
        type=int,
        default=4,
        help="Number of independent chains to sample",
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
    return parser


def set_up_output_directory(args, experiment_name, dir_prefix=None):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    algorithm_subtype = (
        f"{args.metric_type}_metric"
        if args.algorithm == "hmc"
        else f"{args.projection_solver}_solver"
    )
    dir_name = f"{args.algorithm}_{algorithm_subtype}_{timestamp}"
    if dir_prefix is not None:
        dir_name = f"{dir_prefix}_{dir_name}"
    output_dir = os.path.join(args.output_root_dir, experiment_name, dir_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    return output_dir


def set_up_logger(output_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.addHandler(logging.FileHandler(os.path.join(output_dir, "info.log")))
    return logger


def set_up_mici_objects(
    args,
    rng,
    constrained_system_class,
    neg_log_dens_hmc,
    grad_neg_log_dens_hmc,
    constrained_system_kwargs,
):
    if args.algorithm == "chmc":
        system, integrator, adapters, monitor_stats = _set_up_chmc_mici_objects(
            args, constrained_system_class, constrained_system_kwargs
        )
    else:
        system, integrator, adapters, monitor_stats = _set_up_hmc_mici_objects(
            args, neg_log_dens_hmc, grad_neg_log_dens_hmc
        )
    step_size_adapter = mici.adapters.DualAveragingStepSizeAdapter(
        adapt_stat_target=args.step_size_adaptation_target,
        log_step_size_reg_coefficient=args.step_size_reg_coefficient,
    )
    adapters.append(step_size_adapter)
    sampler = mici.samplers.DynamicMultinomialHMC(system, integrator, rng)
    return system, integrator, sampler, adapters, monitor_stats


def _set_up_chmc_mici_objects(
    args, constrained_system_class, constrained_system_kwargs
):
    system = constrained_system_class(**constrained_system_kwargs)
    projection_solver = (
        mlift.jitted_solve_projection_onto_manifold_newton
        if args.projection_solver == "newton"
        else mlift.jitted_solve_projection_onto_manifold_quasi_newton
    )
    projection_solver_kwargs = {
        "constraint_tol": args.projection_solver_warm_up_constraint_tol,
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
    return system, integrator, adapters, monitor_stats


def _set_up_hmc_mici_objects(args, neg_log_dens_hmc, grad_neg_log_dens_hmc):
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
    return system, integrator, adapters, monitor_stats


def precompile_jax_functions(q, args, system):
    start_time = time.time()
    if args.algorithm == "chmc":
        system.precompile_jax_functions(q)
    else:
        system._neg_log_dens(q)
        system._grad_neg_log_dens(q)
    compile_time = time.time() - start_time
    return compile_time


def sample_chains(
    args, sampler, init_states, trace_funcs, adapters, output_dir, monitor_stats
):
    start_time = time.time()
    final_states, traces, stats = sampler.sample_chains_with_adaptive_warm_up(
        args.num_warm_up_iter,
        args.num_main_iter,
        init_states,
        trace_funcs=trace_funcs,
        adapters=adapters,
        memmap_enabled=True,
        memmap_path=output_dir,
        monitor_stats=monitor_stats,
        max_threads_per_process=1,
    )
    sampling_time = time.time() - start_time
    with open(os.path.join(output_dir, "final_chain_states.pkl"), mode="w+b") as f:
        pickle.dump(final_states, f)
    return final_states, traces, stats, sampling_time


def compute_and_print_operation_times(system, final_states, num_call=2000):
    chain_times = []
    chain_func_times = []
    av_times_per_func_call = {}
    state = final_states[0]
    for (qualified_func_name, _), call_count in state._call_counts.items():
        _, func_name = qualified_func_name.split(".")
        if func_name in ["lmult_by_jacob_constr", "normal_space_component"]:
            method = getattr(system, func_name)
            vct = state.pos
            func = lambda: method(state, vct)
        elif func_name in ["rmult_by_jacob_constr", "lmult_by_pinv_jacob_constr"]:
            method = getattr(system, func_name)
            vct = system.constr(state)
            func = lambda: method(state, vct)
        elif func_name == "lmult_by_inv_jacob_product":
            vct = system.constr(state)
            func = lambda: system.lmult_by_inv_jacob_product(state, state, vct)
        else:
            method = getattr(system, func_name).__wrapped__
            func = lambda: method(system, state)
        av_times_per_func_call[func_name] = (
            timeit.timeit(func, number=num_call) / num_call
        )
    for c, state in enumerate(final_states):
        print("-" * 80 + f"\nChain {c}\n" + "-" * 80)
        chain_time = 0
        func_times = {}
        for (qualified_func_name, _), call_count in state._call_counts.items():
            _, func_name = qualified_func_name.split(".")
            av_time_per_call = av_times_per_func_call[func_name]
            func_times[func_name] = call_count * av_time_per_call
            print(
                f"  {func_name}: {call_count} calls × {av_time_per_call:.3g}"
                f" seconds/call = {func_times[func_name]:.3g} seconds"
            )
            chain_time += func_times[func_name]
        print("-" * 80)
        print(f"Total chain operation time: {chain_time:.3g} seconds")
        chain_times.append(chain_time)
        chain_func_times.append(func_times)
    print("=" * 80)
    total_time = sum(chain_times)
    print(f"Total operation time: {total_time:.3g} seconds")
    return {
        "total_operation_time": total_time,
        "per_chain_total_operation_times": chain_times,
        "per_chain_operation_times": chain_func_times,
        "averge_times_per_operation": av_times_per_func_call,
    }


def compute_and_save_summary(output_dir, var_names, traces, **kwargs):
    summary = arviz.summary(traces, var_names=var_names)
    summary_dict = summary.to_dict()
    summary_dict.update(kwargs)
    for key, value in traces.items():
        if key[-6:] == "_calls":
            summary_dict["per_chain_" + key] = [int(v[-1]) for v in value]
    with open(os.path.join(output_dir, "summary.json"), mode="w") as f:
        json.dump(summary_dict, f, ensure_ascii=False, indent=2)
    return summary, summary_dict


def run_experiment(
    args,
    data,
    dim_u,
    rng,
    experiment_name,
    var_names,
    var_trace_func,
    posterior_neg_log_dens,
    constrained_system_class,
    constrained_system_kwargs,
    sample_initial_states,
    dir_prefix=None,
):
    # Set up output directory and logger

    output_dir = set_up_output_directory(args, experiment_name, dir_prefix)
    set_up_logger(output_dir)

    # Set up Mici objects
    (
        neg_log_dens_hmc,
        grad_neg_log_dens_hmc,
    ) = mlift.construct_euclidean_metric_system_functions(
        lambda u: posterior_neg_log_dens(u, data)
    )

    system, integrator, sampler, adapters, monitor_stats = set_up_mici_objects(
        args,
        rng,
        constrained_system_class,
        neg_log_dens_hmc,
        grad_neg_log_dens_hmc,
        constrained_system_kwargs,
    )

    def hamiltonian_and_call_count_trace_func(state):
        h = system.h(state)
        call_counts = {
            name.split(".")[-1] + "_calls": val
            for (name, _), val in state._call_counts.items()
        }
        return {**call_counts, "hamiltonian": h}

    trace_funcs = [var_trace_func, hamiltonian_and_call_count_trace_func]

    # Initialise chain states

    print("Sampling initial states ...")
    init_states = sample_initial_states(rng, args, data)
    init_states = [
        mici.states.ChainState(pos=q, mom=None, dir=1, _call_counts={})
        for q in init_states
    ]

    # Precompile JAX functions to avoid compilation time appearing in chain run times

    print("Pre-compiling JAX functions ...")
    compile_time = precompile_jax_functions(init_states[0].pos, args, system)
    print(f"Total compile time: {compile_time:.0f} seconds")

    # Sample chains

    # Ignore NumPy floating point overflow warnings
    # Prevents warning messages being produced while progress bars are being printed

    np.seterr(over="ignore")

    final_states, traces, stats, sampling_time = sample_chains(
        args, sampler, init_states, trace_funcs, adapters, output_dir, monitor_stats
    )

    print(f"Integrator step size: {integrator.step_size:.2g}")
    print(f"Total sampling time: {sampling_time:.0f} seconds")

    print("Computing chain operation times ...")
    operation_times = compute_and_print_operation_times(system, final_states)

    summary_vars = var_names + ["hamiltonian"]
    summary, summary_dict = compute_and_save_summary(
        output_dir,
        summary_vars,
        traces,
        total_compile_time=compile_time,
        total_sampling_time=sampling_time,
        final_integrator_step_size=integrator.step_size,
        **operation_times,
    )

    print(summary)

    return final_states, traces, stats, summary_dict
