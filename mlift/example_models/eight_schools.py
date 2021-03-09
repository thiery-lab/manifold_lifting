"""Hierarchical model for 'eight-schools' data (Rubin, 1981).

Model definition and data taken from:

https://github.com/stan-dev/stat_comp_benchmarks/tree/master/benchmarks/eight_schools

References:

    Rubin, D. (1981). Estimation in Parallel Randomized Experiments.
    Journal of Educational Statistics, 6(4), 377-401. doi:10.2307/1164617
"""

import os
import numpy as onp
import jax.config
import jax.numpy as np
import jax.lax as lax
import jax.api as api
from mlift.systems import HierarchicalLatentVariableModelSystem
from mlift.distributions import normal, half_cauchy
from mlift.prior import PriorSpecification, set_up_prior
import mlift.example_models.utils as utils

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


prior_specifications = {
    "μ": PriorSpecification(distribution=normal(0, 5)),
    "τ": PriorSpecification(distribution=half_cauchy(5)),
}

compute_dim_u, generate_params, prior_neg_log_dens, sample_from_prior = set_up_prior(
    prior_specifications
)


def generate_from_model(u, v, data):
    params = generate_params(u, data)
    x = params["μ"] + params["τ"] * v
    return params, x


def generate_y(u, v, n, data):
    _, x = generate_from_model(u, v, data)
    return x + data["σ"] * n


def extended_prior_neg_log_dens(q, data):
    dim_u = compute_dim_u(data)
    dim_y = data["y_obs"].shape[0]
    u, v, n = q[:dim_u], q[dim_u : dim_u + dim_y], q[dim_u + dim_y :]
    return prior_neg_log_dens(u, data) + (v ** 2).sum() / 2 + (n ** 2).sum() / 2


def posterior_neg_log_dens(q, data):
    dim_u = compute_dim_u(data)
    u, v = q[:dim_u], q[dim_u:]
    _, x = generate_from_model(u, v, data)
    return (
        prior_neg_log_dens(u, data)
        + (v ** 2).sum() / 2
        + (((data["y_obs"] - x) / data["σ"]) ** 2 / 2 + np.log(data["σ"])).sum()
    )


def sample_initial_states(rng, data, num_chain=4, algorithm="chmc"):
    """Sample initial states from prior."""
    init_states = []
    dim_y = data["y_obs"].shape[0]
    for _ in range(num_chain):
        u = sample_from_prior(rng, data)
        v = rng.standard_normal(dim_y)
        if algorithm == "chmc":
            _, x = generate_from_model(u, v, data)
            n = (data["y_obs"] - x) / data["σ"]
            q = onp.concatenate((u, v, onp.asarray(n)))
        else:
            q = onp.concatenate((u, v))
        init_states.append(q)
    return init_states


if __name__ == "__main__":

    # Process command line arguments defining experiment parameters

    parser = utils.set_up_argparser_with_standard_arguments(
        "Run eight-school hierarchical model experiment"
    )
    args = parser.parse_args()

    # Load data

    data = dict(np.load(os.path.join(args.data_dir, "eight-schools-data.npz")))
    dim_u = compute_dim_u(data)
    dim_y = data["y_obs"].shape[0]

    # Set up seeded random number generator

    rng = onp.random.default_rng(args.seed)

    # Define variables to be traced

    jitted_generate_from_model = api.jit(api.partial(generate_from_model, data=data))

    def trace_func(state):
        u, v = state.pos[:dim_u], state.pos[dim_u : dim_u + dim_y]
        params, x = jitted_generate_from_model(u, v)
        return {**params, "x": x, "u": u, "v": v}

    # Run experiment

    final_states, traces, stats, summary_dict, sampler = utils.run_experiment(
        args=args,
        data=data,
        rng=rng,
        experiment_name="eight_schools",
        var_names=list(prior_specifications.keys()) + ["x"],
        var_trace_func=trace_func,
        posterior_neg_log_dens=posterior_neg_log_dens,
        extended_prior_neg_log_dens=extended_prior_neg_log_dens,
        constrained_system_class=HierarchicalLatentVariableModelSystem,
        constrained_system_kwargs={
            "generate_y": generate_y,
            "data": data,
            "dim_u": dim_u,
        },
        sample_initial_states=sample_initial_states,
    )

