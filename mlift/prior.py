from collections import namedtuple
import numpy as np
from mlift.distributions import normal, pullback_distribution
from mlift.transforms import (
    unbounded_to_lower_bounded,
    unbounded_to_upper_bounded,
    unbounded_to_lower_and_upper_bounded,
)

PriorSpecification = namedtuple(
    "PriorSpecification",
    ("shape", "distribution", "transform"),
    defaults=((), normal(0, 1), None),
)


def reparametrize_to_unbounded_support(prior_spec):
    if (
        prior_spec.distribution.support.lower != -np.inf
        and prior_spec.distribution.support.upper != np.inf
    ):
        bounding_transform = unbounded_to_lower_and_upper_bounded(
            prior_spec.distribution.support.lower, prior_spec.distribution.support.upper
        )
    elif prior_spec.distribution.support.lower != -np.inf:
        bounding_transform = unbounded_to_lower_bounded(
            prior_spec.distribution.support.lower
        )
    elif prior_spec.distribution.support.upper != np.inf:
        bounding_transform = unbounded_to_upper_bounded(
            prior_spec.distribution.support.upper
        )
    else:
        return prior_spec
    distribution = pullback_distribution(prior_spec.distribution, bounding_transform)
    if prior_spec.transform is not None:
        transform = lambda u: prior_spec.transform(bounding_transform(u))
    else:
        transform = bounding_transform
    return PriorSpecification(
        shape=prior_spec.shape, distribution=distribution, transform=transform
    )


def reparametrize_to_standard_normal(prior_spec):
    from_standard_normal_transform = (
        prior_spec.distribution.from_standard_normal_transform
    )
    if prior_spec.transform is not None:
        transform = lambda u: prior_spec.transform(from_standard_normal_transform(u))
    else:
        transform = from_standard_normal_transform
    return PriorSpecification(
        shape=prior_spec.shape, distribution=normal(0, 1), transform=transform
    )


def set_up_prior(prior_specs):
    def get_shape(spec, data):
        return spec.shape(data) if callable(spec.shape) else spec.shape

    def reparametrized_prior_specs(data):
        for name, spec in prior_specs.items():
            if (
                data.get("parametrization") == "normal"
                and spec.distribution.from_standard_normal_transform is not None
            ):
                yield name, reparametrize_to_standard_normal(spec)
            else:
                yield name, reparametrize_to_unbounded_support(spec)

    def reparametrized_prior_specs_and_u_slices(u, data):
        i = 0
        for name, spec in reparametrized_prior_specs(data):
            shape = get_shape(spec, data)
            size = int(np.product(shape))
            u_slice = u[i] if shape == () else u[i : i + size].reshape(shape)
            i += size
            yield name, spec, u_slice

    def compute_dim_u(data):
        return sum(
            int(np.product(get_shape(spec, data)))
            for _, spec in reparametrized_prior_specs(data)
        )

    def generate_params(u, data):
        params = {}
        for name, spec, u_slice in reparametrized_prior_specs_and_u_slices(u, data):
            if spec.transform is not None:
                params[name] = spec.transform(u_slice)
            else:
                params[name] = u_slice
        return params

    def prior_neg_log_dens(u, data):
        nld = 0
        for _, spec, u_slice in reparametrized_prior_specs_and_u_slices(u, data):
            nld += spec.distribution.neg_log_dens(u_slice)
        return nld

    def sample_from_prior(rng, data, num_sample=None):
        u_slices = []
        for _, spec in reparametrized_prior_specs(data):
            shape = get_shape(spec, data)
            if num_sample is None:
                u_slices.append(
                    np.atleast_1d(spec.distribution.sample(rng, shape).flatten())
                )
            else:
                shape = (num_sample,) + shape
                u_slices.append(
                    np.atleast_2d(spec.distribution.sample(rng, shape).reshape((num_sample, -1)))
                )

        return np.concatenate(u_slices, -1)

    return compute_dim_u, generate_params, prior_neg_log_dens, sample_from_prior
