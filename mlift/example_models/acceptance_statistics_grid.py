import argparse
import datetime
import json
import os
import warnings
from functools import partial
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import mici
from mlift.example_models import toy_2d_model
from mlift.transitions import (
    PositionDependentRWMTransition,
    SimplePositionDependentMALATransition,
    XifaraPositionDependentMALATransition,
)


def construct_sampler_getter_functions(model, n_step=10):
    def get_rwm_sampler(sigma, step_size, rng):
        system = mici.systems.EuclideanMetricSystem(
            neg_log_dens=partial(model.neg_log_posterior_dens, sigma=sigma, y=model.y),
            grad_neg_log_dens=lambda q: q * 0,
        )
        integrator = mici.integrators.LeapfrogIntegrator(system, step_size=step_size)
        return mici.samplers.StaticMetropolisHMC(system, integrator, rng, n_step=1)

    def get_mala_sampler(sigma, step_size, rng):
        system = mici.systems.EuclideanMetricSystem(
            neg_log_dens=partial(model.neg_log_posterior_dens, sigma=sigma, y=model.y),
            grad_neg_log_dens=partial(
                model.grad_neg_log_posterior_dens, sigma=sigma, y=model.y
            ),
        )
        integrator = mici.integrators.LeapfrogIntegrator(system, step_size=step_size)
        return mici.samplers.StaticMetropolisHMC(system, integrator, rng, n_step=1)

    def get_hmc_sampler(sigma, step_size, rng):
        system = mici.systems.EuclideanMetricSystem(
            neg_log_dens=partial(model.neg_log_posterior_dens, sigma=sigma, y=model.y),
            grad_neg_log_dens=partial(
                model.grad_neg_log_posterior_dens, sigma=sigma, y=model.y
            ),
        )
        integrator = mici.integrators.LeapfrogIntegrator(system, step_size=step_size)
        return mici.samplers.StaticMetropolisHMC(system, integrator, rng, n_step=n_step)

    def get_fisher_rmhmc_sampler(sigma, step_size, rng):
        system = mici.systems.DenseRiemannianMetricSystem(
            neg_log_dens=partial(model.neg_log_posterior_dens, sigma=sigma, y=model.y),
            grad_neg_log_dens=partial(
                model.grad_neg_log_posterior_dens, sigma=sigma, y=model.y
            ),
            metric_func=partial(model.metric, sigma=sigma),
            vjp_metric_func=partial(model.vjp_metric, sigma=sigma),
        )
        integrator = mici.integrators.ImplicitLeapfrogIntegrator(
            system, step_size=step_size
        )
        return mici.samplers.StaticMetropolisHMC(system, integrator, rng, n_step=n_step)

    def get_softabs_rmhmc_sampler(sigma, step_size, rng):
        system = mici.systems.SoftAbsRiemannianMetricSystem(
            neg_log_dens=partial(model.neg_log_posterior_dens, sigma=sigma, y=model.y),
            grad_neg_log_dens=partial(
                model.grad_neg_log_posterior_dens, sigma=sigma, y=model.y
            ),
            hess_neg_log_dens=partial(
                model.hess_neg_log_posterior_dens, sigma=sigma, y=model.y
            ),
            mtp_neg_log_dens=partial(
                model.mtp_neg_log_posterior_dens, sigma=sigma, y=model.y
            ),
            softabs_coeff=1.0,
        )
        integrator = mici.integrators.ImplicitLeapfrogIntegrator(
            system, step_size=step_size
        )
        return mici.samplers.StaticMetropolisHMC(system, integrator, rng, n_step=n_step)

    def get_chmc_sampler(sigma, step_size, rng):
        system = mici.systems.DenseConstrainedEuclideanMetricSystem(
            neg_log_dens=model.neg_log_prior_dens,
            grad_neg_log_dens=model.grad_neg_log_prior_dens,
            dens_wrt_hausdorff=False,
            constr=partial(model.constr, sigma=sigma, y=model.y),
            jacob_constr=partial(model.jacob_constr, sigma=sigma, y=model.y),
            mhp_constr=partial(model.mhp_constr, sigma=sigma, y=model.y),
        )
        integrator = mici.integrators.ConstrainedLeapfrogIntegrator(
            system,
            step_size=step_size,
            projection_solver=mici.solvers.solve_projection_onto_manifold_newton,
        )
        return mici.samplers.StaticMetropolisHMC(system, integrator, rng, n_step=n_step)

    def get_pd_rwm_sampler(sigma, step_size, rng):
        transition = PositionDependentRWMTransition(
            neg_log_dens=partial(model.neg_log_posterior_dens, sigma=sigma, y=model.y),
            metric=partial(model.metric, sigma=sigma),
            step_size=step_size,
        )
        return mici.samplers.MarkovChainMonteCarloMethod(rng, {"default": transition})

    def get_simple_pd_mala_sampler(sigma, step_size, rng):
        transition = SimplePositionDependentMALATransition(
            neg_log_dens=partial(model.neg_log_posterior_dens, sigma=sigma, y=model.y),
            grad_neg_log_dens=partial(
                model.grad_neg_log_posterior_dens, sigma=sigma, y=model.y
            ),
            metric=partial(model.metric, sigma=sigma),
            step_size=step_size,
        )
        return mici.samplers.MarkovChainMonteCarloMethod(rng, {"default": transition})

    def get_xifara_pd_mala_sampler(sigma, step_size, rng):
        transition = XifaraPositionDependentMALATransition(
            neg_log_dens=partial(model.neg_log_posterior_dens, sigma=sigma, y=model.y),
            grad_neg_log_dens=partial(
                model.grad_neg_log_posterior_dens, sigma=sigma, y=model.y
            ),
            metric=partial(model.metric, sigma=sigma),
            jacob_metric=partial(model.jacob_metric, sigma=sigma),
            step_size=step_size,
        )
        return mici.samplers.MarkovChainMonteCarloMethod(rng, {"default": transition})

    def get_initial_states_chmc(theta_inits, sigma):
        return [
            mici.states.ChainState(
                pos=np.concatenate(
                    (theta, (model.y - model.forward_func(theta)) / sigma)
                ),
                mom=None,
                dir=1,
            )
            for theta in theta_inits
        ]

    def get_initial_states_default(theta_inits, sigma):
        return [
            mici.states.ChainState(pos=theta, mom=None, dir=1) for theta in theta_inits
        ]

    return {
        "rwm": (get_rwm_sampler, get_initial_states_default),
        "mala": (get_mala_sampler, get_initial_states_default),
        "hmc": (get_hmc_sampler, get_initial_states_default),
        "fisher_rmhmc": (get_fisher_rmhmc_sampler, get_initial_states_default),
        "softabs_rmhmc": (get_softabs_rmhmc_sampler, get_initial_states_default),
        "chmc": (get_chmc_sampler, get_initial_states_chmc),
        "pd_rwm": (get_pd_rwm_sampler, get_initial_states_default),
        "simple_pd_mala": (get_simple_pd_mala_sampler, get_initial_states_default),
        "xifara_pd_mala": (get_xifara_pd_mala_sampler, get_initial_states_default),
    }


def compute_acceptance_statistic_grid(
    obs_noise_std_grid,
    step_size_grid,
    theta_inits,
    n_sample,
    seed,
    get_sampler,
    get_initial_states,
):
    rng = np.random.default_rng(seed)
    average_accept_stats = np.full(
        (len(obs_noise_std_grid), len(step_size_grid)), np.nan
    )
    with mici.progressbars.ProgressBar(
        list(enumerate(product(obs_noise_std_grid, step_size_grid))),
        "Computing average acceptance statistics",
        unit="(σ, ϵ) pair",
    ) as pb:
        for (i, (obs_noise_std, step_size)), _ in pb:
            sampler = get_sampler(obs_noise_std, step_size, rng)
            init_states = get_initial_states(theta_inits, obs_noise_std)
            _, _, stats = sampler.sample_chains(
                n_sample,
                init_states,
                trace_funcs=[],
                n_process=len(theta_inits),
                display_progress=False,
            )
            if "default" in stats:
                stats = stats["default"]
            average_accept_stats.flat[i] = np.concatenate(stats["accept_stat"]).mean()
    return average_accept_stats


def plot_acceptance_statistics_heatmap(
    ax, obs_noise_std_grid, step_size_grid, average_accept_stats
):
    artist = ax.imshow(average_accept_stats.T, vmin=0, vmax=1, origin="lower")
    ax.set(
        xlabel="Noise scale $\sigma$",
        ylabel="Step size $\epsilon$",
        aspect=1,
        xticks=np.arange(obs_noise_std_grid.shape[0])[::2],
        xticklabels=[f"$10^{{{int(i)}}}$" for i in np.log10(obs_noise_std_grid[::2])],
        yticks=np.arange(step_size_grid.shape[0])[::2],
        yticklabels=[f"$10^{{{int(i)}}}$" for i in np.log10(step_size_grid[::2])],
    )
    return artist


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Run acceptance rate grid experiment for toy two-dimensional loop model",
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
        choices=(
            "rwm",
            "mala",
            "hmc",
            "fisher_rmhmc",
            "softabs_rmhmc",
            "chmc",
            "pd_rwm",
            "simple_pd_mala",
            "xifara_pd_mala",
        ),
        help="Which algorithm to construct acceptance rate grid for",
    )
    parser.add_argument(
        "--seed", type=int, default=202101, help="Seed for random number generator"
    )
    parser.add_argument(
        "--obs-noise-std-logspace-range",
        type=int,
        nargs=2,
        default=[-5, 0],
        help=(
            "Range of exponents of log spaced grid of observation noise scales to "
            "compute acceptance statistics over. Grid ranges from 10 to power of first "
            "value to 10 to power of second value."
        ),
    )
    parser.add_argument(
        "--step-size-logspace-range",
        type=int,
        nargs=2,
        default=[-5, 0],
        help=(
            "Range of exponents of log spaced grid of step sizes to compute acceptance "
            "statistics over. Grid ranges from 10 to power of first value to 10 to "
            "power of second value."
        ),
    )
    parser.add_argument(
        "--num-chain",
        type=int,
        default=4,
        help="Number of independent chains to sample",
    )
    parser.add_argument(
        "--num-iter", type=int, default=1000, help="Number of iterations per chain",
    )
    args = parser.parse_args()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    dir_name = f"{args.algorithm}_seed_{args.seed}_{timestamp}"
    output_dir = os.path.join(
        args.output_root_dir, "toy_2d_model_acceptance_statistics", dir_name
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    get_sampler, get_initial_states = construct_sampler_getter_functions(toy_2d_model)[
        args.algorithm
    ]
    theta_inits = np.concatenate(
        toy_2d_model.solve_for_limiting_manifold(toy_2d_model.y, args.num_chain)
    )
    obs_noise_std_grid = np.logspace(
        *args.obs_noise_std_logspace_range,
        2
        * (args.obs_noise_std_logspace_range[1] - args.obs_noise_std_logspace_range[0])
        + 1,
    )
    step_size_grid = np.logspace(
        *args.step_size_logspace_range,
        2 * (args.step_size_logspace_range[1] - args.step_size_logspace_range[0]) + 1,
    )
    # Disable runtime warnings to prevent interference with progress meter display
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    average_acceptance_stats = compute_acceptance_statistic_grid(
        obs_noise_std_grid,
        step_size_grid,
        theta_inits,
        args.num_iter,
        args.seed,
        get_sampler,
        get_initial_states,
    )
    np.savez(
        os.path.join(output_dir, "acceptance_statistics.npz"),
        step_size_grid=obs_noise_std_grid,
        obs_noise_std_grid=step_size_grid,
        average_acceptance_stats=average_acceptance_stats,
    )
    with plt.style.context(
        {
            "font.family": "Latin Modern Roman",
            "mathtext.fontset": "cm",
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 10,
            "axes.linewidth": 0.5,
            "lines.linewidth": 1.0,
            "axes.labelpad": 2.0,
        }
    ):
        fig, ax = plt.subplots(figsize=(5, 4))
        artist = plot_acceptance_statistics_heatmap(
            ax, obs_noise_std_grid, step_size_grid, average_acceptance_stats
        )
        fig.colorbar(artist, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "acceptance_statistics_heatmap.pdf"))

