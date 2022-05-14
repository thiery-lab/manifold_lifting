"""Toy two-dimensional example model.

Toy model with forward function

    F(theta) = theta[1] ** 2 + theta[0]**2 * (theta[0]**2 - 0.5)

and observation model

    y = F(theta) + sigma * eta

where eta ~ normal(0, 1) and sigma > 0 is fixed.
"""

import argparse
import datetime
import json
import os
import time
import warnings
from functools import partial
import arviz
import numpy as np
import mici
import symnum.numpy as snp
from symnum import (
    numpify,
    jacobian,
    grad,
    hessian,
    vector_jacobian_product,
    matrix_hessian_product,
    matrix_tressian_product,
)


dim_theta = 2
dim_y = 1
y = 1


@numpify(dim_theta)
def forward_func(theta):
    return snp.array([theta[1] ** 2 + theta[0] ** 2 * (theta[0] ** 2 - 0.5)])


@numpify(dim_theta, None, None)
def neg_log_posterior_dens(theta, sigma, y):
    return (
        snp.sum(theta ** 2, 0) + snp.sum((y - forward_func(theta)) ** 2, 0) / sigma ** 2
    ) / 2


grad_neg_log_posterior_dens = grad(neg_log_posterior_dens, return_aux=True)
hess_neg_log_posterior_dens = hessian(neg_log_posterior_dens, return_aux=True)
mtp_neg_log_posterior_dens = matrix_tressian_product(
    neg_log_posterior_dens, return_aux=True
)


@numpify(dim_theta, None)
def metric(theta, sigma):
    jac = jacobian(forward_func)(theta)
    return jac.T @ jac / sigma ** 2 + snp.identity(dim_theta)


vjp_metric = vector_jacobian_product(metric, return_aux=True)


@numpify(dim_theta + dim_y)
def neg_log_prior_dens(q):
    return snp.sum(q ** 2) / 2


grad_neg_log_prior_dens = grad(neg_log_prior_dens, return_aux=True)


@numpify(dim_theta + dim_y, None, None)
def constr(q, sigma, y):
    theta, eta = q[:dim_theta], q[dim_theta:]
    return forward_func(theta) + sigma * eta - y


jacob_constr = jacobian(constr, return_aux=True)
mhp_constr = matrix_hessian_product(constr, return_aux=True)


@numpify(dim_theta, None, None, None)
def neg_log_lifted_posterior_dens(theta, eta, sigma, y):
    jac = jacobian(forward_func)(theta)
    return (
        snp.sum(theta ** 2, 0) / 2
        + eta ** 2 / 2
        + snp.log(jac @ jac.T + sigma ** 2)[0, 0] / 2
    )


def get_constrained_hmc_mici_objects(args, theta_inits, sigma):
    init_states = [
        mici.states.ChainState(
            pos=np.concatenate([theta, (y - forward_func(theta)) / sigma]),
            mom=None,
            dir=1,
            _call_counts={},
        )
        for theta in theta_inits
    ]
    projection_solver = (
        mici.solvers.solve_projection_onto_manifold_newton
        if args.projection_solver == "newton"
        else mici.solvers.solve_projection_onto_manifold_quasi_newton
    )
    system = mici.systems.DenseConstrainedEuclideanMetricSystem(
        neg_log_dens=neg_log_prior_dens,
        grad_neg_log_dens=grad_neg_log_prior_dens,
        dens_wrt_hausdorff=False,
        constr=partial(constr, sigma=sigma, y=y),
        jacob_constr=partial(jacob_constr, sigma=sigma, y=y),
        mhp_constr=partial(mhp_constr, sigma=sigma, y=y),
    )
    integrator = mici.integrators.ConstrainedLeapfrogIntegrator(
        system, projection_solver=projection_solver
    )
    adapters = [
        mici.adapters.DualAveragingStepSizeAdapter(
            args.step_size_adaptation_target,
            log_step_size_reg_coefficient=args.step_size_reg_coefficient,
        )
    ]
    return init_states, system, integrator, adapters


def get_standard_hmc_mici_objects(args, theta_inits, sigma):
    init_states = [
        mici.states.ChainState(pos=theta, mom=None, dir=1, _call_counts={})
        for theta in theta_inits
    ]
    system = mici.systems.EuclideanMetricSystem(
        neg_log_dens=partial(neg_log_posterior_dens, sigma=sigma, y=y),
        grad_neg_log_dens=partial(grad_neg_log_posterior_dens, sigma=sigma, y=y),
    )
    integrator = mici.integrators.LeapfrogIntegrator(system)
    step_size_adapter = mici.adapters.DualAveragingStepSizeAdapter(
        args.step_size_adaptation_target,
        log_step_size_reg_coefficient=args.step_size_reg_coefficient,
    )
    metric_adapter = (
        mici.adapters.OnlineCovarianceMetricAdapter()
        if args.hmc_metric_type == "dense"
        else mici.adapters.OnlineVarianceMetricAdapter()
    )
    adapters = [step_size_adapter, metric_adapter]
    return init_states, system, integrator, adapters


def get_riemannian_manifold_hmc_mici_objects(args, theta_inits, sigma):
    init_states = [
        mici.states.ChainState(pos=theta, mom=None, dir=1, _call_counts={})
        for theta in theta_inits
    ]
    if args.rmhmc_metric_type == "softabs":
        system = mici.systems.SoftAbsRiemannianMetricSystem(
            neg_log_dens=partial(neg_log_posterior_dens, sigma=sigma, y=y),
            grad_neg_log_dens=partial(grad_neg_log_posterior_dens, sigma=sigma, y=y),
            hess_neg_log_dens=partial(hess_neg_log_posterior_dens, sigma=sigma, y=y),
            mtp_neg_log_dens=partial(mtp_neg_log_posterior_dens, sigma=sigma, y=y),
            softabs_coeff=args.softabs_coefficient,
        )
    else:
        system = mici.systems.DenseRiemannianMetricSystem(
            neg_log_dens=partial(neg_log_posterior_dens, sigma=sigma, y=y),
            grad_neg_log_dens=partial(grad_neg_log_posterior_dens, sigma=sigma, y=y),
            metric_func=partial(metric, sigma=sigma),
            vjp_metric_func=partial(vjp_metric, sigma=sigma),
        )
    fixed_point_solver = (
        mici.solvers.solve_fixed_point_direct
        if args.projection_solver == "direct"
        else mici.solvers.solve_fixed_point_steffensen
    )
    integrator = mici.integrators.ImplicitLeapfrogIntegrator(
        system, fixed_point_solver=fixed_point_solver,
    )
    adapters = [
        mici.adapters.DualAveragingStepSizeAdapter(
            args.step_size_adaptation_target,
            log_step_size_reg_coefficient=args.step_size_reg_coefficient,
        )
    ]
    return init_states, system, integrator, adapters


def get_trace_func(system):
    def trace_func(state):
        call_counts = {
            name.split(".")[-1] + "_calls": val
            for (name, _), val in state._call_counts.items()
        }
        return {
            "theta": state.pos,
            **call_counts,
            "hamiltonian": system.h(state),
            "neg_log_dens": system.neg_log_dens(state),
        }

    return trace_func


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "Run toy two-dimensional loop model experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-root-dir",
        default="results",
        help="Root directory to make experiment output subdirectory in",
    )
    parser.add_argument(
        "--algorithm",
        default="chmc",
        choices=("chmc", "hmc", "rmhmc"),
        help="Which algorithm to perform inference with, from: chmc, hmc, rmhmc",
    )
    parser.add_argument(
        "--seed", type=int, default=202101, help="Seed for random number generator"
    )
    parser.add_argument(
        "--obs-noise-std",
        type=float,
        default=1.0,
        help="Standard deviation of observation noise to use",
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
        "--max-tree-depth",
        type=int,
        default=15,
        help="Maximum depth of binary trajectory tree in each dynamic HMC iteration",
    )
    parser.add_argument(
        "--step-size-adaptation-target",
        type=float,
        default=0.9,
        help="Target acceptance statistic for step size adaptation",
    )
    parser.add_argument(
        "--step-size-reg-coefficient",
        type=float,
        default=0.1,
        help="Regularisation coefficient for step size adaptation",
    )
    parser.add_argument(
        "--hmc-metric-type",
        choices=("diagonal", "dense"),
        default="diagonal",
        help=(
            "Metric type to adaptively tune during warm-up stage when using HMC "
            "algorithm. If 'diagonal' a diagonal metric matrix representation is used "
            "with diagonal entries set to reciprocals of estimates of the marginal "
            "posterior variances. If 'dense' a dense metric matrix representation is "
            "used corresponding to the inverse of an estimate of the posterior "
            "covariance matrix."
        ),
    )
    parser.add_argument(
        "--rmhmc-metric-type",
        choices=("fisher", "softabs"),
        default="fisher",
        help=(
            "Metric type to use when using RM-HMC algorithm. If 'fisher', the empirical"
            " Fisher information matrix is used as the metric. If 'softabs' the Hessian"
            " of the negative log posterior density is used with eigenvalues mapped to "
            "strictly positive values using a 'SoftAbs' smooth approximation to the "
            "absolute function."
        ),
    )
    parser.add_argument(
        "--softabs-coefficient",
        type=float,
        default=1.0,
        help=(
            "Positive coefficient to use in SoftAbs function when using RM-HMC with "
            "SoftAbs metric. As the coefficient tends to infinity the SoftAbs function "
            "tends to the absolute function."
        ),
    )
    parser.add_argument(
        "--projection-solver",
        choices=("newton", "quasi-newton"),
        default="newton",
        help=(
            "Iterative method to solve projection onto manifold when using C-HMC "
            "algorithm."
        ),
    )
    parser.add_argument(
        "--fixed-point-solver",
        choices=("direct", "steffensen"),
        default="direct",
        help=(
            "Iterative method to solve fixed point equations when using RM-HMC "
            "algorithm."
        ),
    )
    parser.add_argument(
        "--run-chmc-to-initialise",
        action="store_true",
        help=(
            "Run short C-HMC chains from prior samples and use final states as initial "
            "states in main runs."
        ),
    )
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    get_algorithm_mici_objects = {
        "chmc": get_constrained_hmc_mici_objects,
        "rmhmc": get_riemannian_manifold_hmc_mici_objects,
        "hmc": get_standard_hmc_mici_objects,
    }[args.algorithm]
    algorithm_subtype = {
        "chmc": args.projection_solver,
        "rmhmc": args.rmhmc_metric_type,
        "hmc": args.hmc_metric_type,
    }[args.algorithm]
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    dir_name = (
        f"σ_{args.obs_noise_std:.0e}_{args.algorithm}_{algorithm_subtype}_"
        f"seed_{args.seed}_{timestamp}"
    )
    output_dir = os.path.join(args.output_root_dir, "toy_2d_model", dir_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    print(
        f"Running experiment with toy 2D model with σ = {args.obs_noise_std} using "
        f"{args.algorithm.upper()} ({algorithm_subtype}) algorithm for inference. \n"
        f"Results will be saved to {output_dir}"
    )
    theta_inits = rng.standard_normal((args.num_chain, dim_theta))
    # Disable runtime warnings to prevent interference with progress meter display
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if args.run_chmc_to_initialise:
        print("Running short CHMC chains to get initial states")
        (
            init_states,
            system,
            integrator,
            adapters,
        ) = get_constrained_hmc_mici_objects(args, theta_inits, args.obs_noise_std)
        sampler = mici.samplers.DynamicMultinomialHMC(system, integrator, rng)
        final_states, _, _ = sampler.sample_chains_with_adaptive_warm_up(
            50, 100, init_states, adapters=adapters, n_process=args.num_chain,
        )
        theta_inits = [state.pos[:2] for state in final_states]
        print(f"Running main {args.algorithm.upper()} ({algorithm_subtype}) chains")
    init_states, system, integrator, adapters = get_algorithm_mici_objects(
        args, theta_inits, args.obs_noise_std
    )
    trace_func = get_trace_func(system)
    sampler = mici.samplers.DynamicMultinomialHMC(
        system, integrator, rng, max_tree_depth=args.max_tree_depth
    )
    start_time = time.time()
    _, traces, _ = sampler.sample_chains_with_adaptive_warm_up(
        args.num_warm_up_iter,
        args.num_main_iter,
        init_states,
        trace_funcs=[trace_func],
        adapters=adapters,
        monitor_stats=["accept_stat", "n_step", "diverging"],
        n_process=args.num_chain,
        memmap_enabled=True,
        memmap_path=output_dir,
    )
    sampling_time = time.time() - start_time
    print(f"Integrator step size: {integrator.step_size:.2g}")
    print(f"Total sampling time: {sampling_time:.0f} seconds")
    summary = arviz.summary(
        traces, var_names=["theta", "hamiltonian", "neg_log_dens"]
    )
    print(summary)
    summary_dict = summary.to_dict()
    summary_dict["total_sampling_time"] = sampling_time
    summary_dict["final_integrator_step_size"] = integrator.step_size
    for key, value in traces.items():
        if key[-6:] == "_calls":
            summary_dict["per_chain_" + key] = [int(v[-1]) for v in value]
    with open(os.path.join(output_dir,  f"summary.json"), "w") as f:
        json.dump(summary_dict, f, ensure_ascii=False, indent=2)
