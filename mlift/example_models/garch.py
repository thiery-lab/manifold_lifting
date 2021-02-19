"""Generalised autoregressive conditional heteroscedastic (GARCH) benchmark model

Model definition and data taken from:

https://github.com/stan-dev/stat_comp_benchmarks/tree/master/benchmarks/garch
"""

import os
import numpy as onp
import jax.config
import jax.numpy as np
import jax.lax as lax
import jax.api as api
from mlift.systems import IndependentAdditiveNoiseModelSystem
from mlift.distributions import normal, uniform, half_cauchy
from mlift.prior import PriorSpecification, set_up_prior
import mlift.example_models.utils as utils

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


prior_specifications = {
    "μ": PriorSpecification(distribution=normal(0, 10)),
    "α_0": PriorSpecification(distribution=half_cauchy(2.5)),
    "α_1": PriorSpecification(distribution=uniform(0, 1)),
    # β_1 ~ uniform(0, 1 - α_1) therefore β_1 / (1 - α_1) ~ uniform(0, 1)
    "β_1_over_1_minus_α_1": PriorSpecification(distribution=uniform(0, 1)),
}

compute_dim_u, generate_params, prior_neg_log_dens, sample_from_prior = set_up_prior(
    prior_specifications
)


def generate_from_model(u, data):
    params = generate_params(u, data)
    params["β_1"] = params.pop("β_1_over_1_minus_α_1") * (1 - params["α_1"])

    def step(x, y):
        x = params["α_0"] + params["α_1"] * (y - params["μ"]) ** 2 + params["β_1"] * x
        return x, x

    _, x_ = lax.scan(step, data["x_0"], data["y_obs"][:-1])

    x = np.concatenate((np.array([data["x_0"]]), x_))

    return params, x


def generate_y(u, n, data):
    params, x = generate_from_model(u, data)
    y = params["μ"] + np.sqrt(x) * n
    return y


def extended_prior_neg_log_dens(q, data):
    dim_u = compute_dim_u(data)
    u, n = q[:dim_u], q[dim_u:]
    return prior_neg_log_dens(u, data) + (n ** 2).sum() / 2


def posterior_neg_log_dens(u, data):
    params, x = generate_from_model(u, data)
    return prior_neg_log_dens(u, data) + (
        ((data["y_obs"] - params["μ"]) ** 2 / x).sum() / 2 + np.log(x).sum() / 2
    )


def sample_initial_states(rng, data, num_chain=4, algorithm="chmc"):
    """Sample initial states from prior."""
    init_states = []
    for _ in range(num_chain):
        u = sample_from_prior(rng, data)
        if algorithm == "chmc":
            params, x = generate_from_model(u, data)
            n = (data["y_obs"] - params["μ"]) / onp.sqrt(x)
            q = onp.concatenate((u, onp.asarray(n)))
        else:
            q = onp.asarray(u)
        init_states.append(q)
    return init_states


if __name__ == "__main__":

    # Process command line arguments defining experiment parameters

    parser = utils.set_up_argparser_with_standard_arguments(
        "Run generalised autoregressive conditional heteroscedasticity (GARCH) "
        "benchmark model experiment"
    )
    args = parser.parse_args()

    # Load data

    data = dict(np.load(os.path.join(args.data_dir, "garch-benchmark-data.npz")))
    dim_u = compute_dim_u(data)

    # Set up seeded random number generator

    rng = onp.random.default_rng(args.seed)

    # Define variables to be traced

    jitted_generate_from_model = api.jit(api.partial(generate_from_model, data=data))

    def trace_func(state):
        u = state.pos[:dim_u]
        params, x = jitted_generate_from_model(u)
        return {**params, "x": x, "u": u}

    # Run experiment

    final_states, traces, stats, summary_dict, sampler = utils.run_experiment(
        args=args,
        data=data,
        dim_u=dim_u,
        rng=rng,
        experiment_name="garch",
        var_names=["μ", "α_0", "α_1", "β_1"],
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

