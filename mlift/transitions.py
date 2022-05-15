"""Markov transitions for position-dependent RWM and MALA schemes."""

import numpy as np
import scipy.linalg as sla
from mici.states import cache_in_state, cache_in_state_with_aux
from mici.transitions import Transition


class PositionDependentRWMTransition(Transition):
    """Markov transition for position-dependent Gaussian random-walk Metropolis scheme.

    References:

    1. Livingstone, S. (2015). Geometric ergodicity of the random walk Metropolis with
       position-dependent proposal covariance. https://doi.org/10.48550/arXiv.1507.05780
    """

    def __init__(self, neg_log_dens, metric, step_size):
        self._neg_log_dens = neg_log_dens
        self._metric = metric
        self.step_size = step_size

    @property
    def state_variables(self):
        return {"pos"}

    @property
    def statistic_types(self):
        return {"accept_stat": (np.float64, np.nan)}

    @cache_in_state("pos")
    def neg_log_dens(self, state):
        return self._neg_log_dens(state.pos)

    @cache_in_state("pos")
    def metric(self, state):
        return self._metric(state.pos)

    @cache_in_state("pos")
    def chol_metric(self, state):
        metric = self.metric(state)
        return np.linalg.cholesky(metric)

    def sample(self, state, rng):
        proposed_state = state.copy()
        noise = self.step_size * sla.solve_triangular(
            self.chol_metric(state).T, rng.standard_normal(state.pos.shape), lower=False
        )
        proposed_state.pos = state.pos + noise
        log_accept_ratio = (
            self.neg_log_dens(state)
            - self.neg_log_dens(proposed_state)
            + noise
            @ (self.metric(state) - self.metric(proposed_state))
            @ noise
            / (2 * self.step_size ** 2)
            + np.log(self.chol_metric(proposed_state).diagonal()).sum()
            - np.log(self.chol_metric(state).diagonal()).sum()
        )
        accept_prob = np.exp(min(0, log_accept_ratio))
        if rng.uniform() < accept_prob:
            state = proposed_state
        return state, {"accept_stat": accept_prob}


class SimplePositionDependentMALATransition(PositionDependentRWMTransition):
    """Markov transition for position-dependent Metropolis-adjusted Langevin algorithm.

    Specifically the 'simplfied' proposal suggested at the end of Section 5 of [1] is
    used here which ignores derivatives in the metric required for this proposal to
    correspond to an Euler-Maruyama discretisation of a Langevin diffusion with the
    target distribution as its stationary distribution [2].

    References:

    1. Girolami, M. and Calderhead, B. (2011) Riemann manifold Langevin and Hamiltonian
       Monte Carlo methods. Journal of the Royal Statistical Society: Series B
       (Statistical Methodology), 73, 123–214.
    2. Xifara, T., Sherlock, C., Livingstone, S., Byrne, S. and Girolami, M. (2014).
       Langevin diffusions and the Metropolis-adjusted Langevin algorithm. Statistics
       & Probability Letters, 91, 14–19.
    """

    def __init__(self, neg_log_dens, grad_neg_log_dens, metric, step_size):
        super().__init__(neg_log_dens, metric, step_size)
        self._grad_neg_log_dens = grad_neg_log_dens

    @cache_in_state_with_aux("pos", "neg_log_dens")
    def grad_neg_log_dens(self, state):
        return self._grad_neg_log_dens(state.pos)

    def grad_potential(self, state):
        return self.grad_neg_log_dens(state)

    def proposal_mean(self, state):
        return (
            state.pos
            - self.step_size ** 2
            * sla.cho_solve((self.chol_metric(state), True), self.grad_potential(state))
            / 2
        )

    def sample(self, state, rng):
        proposed_state = state.copy()
        noise = self.step_size * sla.solve_triangular(
            self.chol_metric(state).T, rng.standard_normal(state.pos.shape), lower=False
        )
        proposal_mean_fwd = self.proposal_mean(state)
        proposed_state.pos = proposal_mean_fwd + noise
        proposal_mean_bwd = self.proposal_mean(proposed_state)
        noise_bwd = state.pos - proposal_mean_bwd
        log_accept_ratio = (
            self.neg_log_dens(state)
            - self.neg_log_dens(proposed_state)
            + noise @ self.metric(state) @ noise / (2 * self.step_size ** 2)
            - noise_bwd
            @ self.metric(proposed_state)
            @ noise_bwd
            / (2 * self.step_size ** 2)
            + np.log(self.chol_metric(proposed_state).diagonal()).sum()
            - np.log(self.chol_metric(state).diagonal()).sum()
        )
        accept_prob = np.exp(min(0, log_accept_ratio))
        if rng.uniform() < accept_prob:
            state = proposed_state
        return state, {"accept_stat": accept_prob}


class XifaraPositionDependentMALATransition(SimplePositionDependentMALATransition):
    """Markov transition for position-dependent Metropolis-adjusted Langevin algorithm.

    Specifically the proposal suggested in Section 3 of [1] is used here which
    corresponds to an Euler-Maruyama discretisation of a Langevin diffusion with the
    target distribution as its stationary distribution.

    References:

    1. Xifara, T., Sherlock, C., Livingstone, S., Byrne, S. and Girolami, M. (2014).
       Langevin diffusions and the Metropolis-adjusted Langevin algorithm. Statistics
       & Probability Letters, 91, 14–19.
    """

    def __init__(
        self, neg_log_dens, grad_neg_log_dens, metric, jacob_metric, step_size
    ):
        super().__init__(neg_log_dens, grad_neg_log_dens, metric, step_size)
        self._jacob_metric = jacob_metric

    @cache_in_state_with_aux("pos", "metric")
    def jacob_metric(self, state):
        return self._jacob_metric(state.pos)

    def grad_potential(self, state):
        jacob_metric = self.jacob_metric(state)
        metric = self.metric(state)
        return self.grad_neg_log_dens(state) + (
            jacob_metric * np.linalg.inv(metric)[None]
        ).sum((1, 2))
