#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.

import logging
import math
from collections import defaultdict
from typing import Dict, Generic, List, Optional, TypeVar

import numpy as np
from scipy.special import loggamma
from scipy.stats import norm


logger = logging.getLogger(__name__)


T = TypeVar("T")


class Counter(Generic[T]):
    def __init__(self) -> None:
        self.counts: Dict[T, int] = defaultdict(int)
        self.count_sum: int = 0

    def __repr__(self):
        return f"counts: {dict(self.counts)}. count_sum = {self.count_sum}."

    def increment(self, observation: T) -> None:
        self.counts[observation] += 1
        self.count_sum += 1

    def decrement(self, observation: T) -> None:
        if observation not in self.counts or self.counts[observation] < 1:
            raise RuntimeError(
                f"Trying to decrement {observation}, but was never observed"
            )

        self.counts[observation] -= 1
        self.count_sum -= 1

    def get_count(self, observation: T) -> int:
        return self.counts[observation]

    def get_count_sum(self) -> int:
        return self.count_sum


class DirichletPrior:
    def __init__(self, dimension: int, scale: float, mean_vals: List[float]) -> None:
        self.dimension = dimension
        self.scale = scale
        self.mean_vals = mean_vals
        self.vals = [self.scale * mean_val for mean_val in self.mean_vals]

        self._validate()

    def __repr__(self):
        return (
            f"dimension = {self.dimension}. "
            f"scale = {self.scale}. "
            f"mean = {self.mean_vals}."
        )

    @classmethod
    def from_dim_scale(cls, dimension: int, scale: float) -> "DirichletPrior":
        prior = cls(dimension, scale, mean_vals=[(1.0 / dimension)] * dimension)
        return prior

    @classmethod
    def from_scale_mean(cls, scale: float, mean_vals: List[float]) -> "DirichletPrior":
        prior = cls(len(mean_vals), scale, mean_vals=mean_vals)
        return prior

    def _validate(self):
        if abs(sum(self.mean_vals) - 1.0) > 1e-6:
            raise RuntimeError(f"Invalid DirichletPrior {self.mean_vals}")


class DirichletMultinomial:
    def __init__(self, prior: DirichletPrior, data: Counter = None) -> None:
        self.prior: DirichletPrior = prior
        self.data: Counter = data if data is not None else Counter()
        self.posteriors: List[List[float]] = []

    def __repr__(self):
        return (
            f"prior: {self.prior}. data: {self.data}. "
            f"posterior: {self.get_posterior_dist()}"
        )

    @classmethod
    def from_prior(cls, prior) -> "DirichletMultinomial":
        return DirichletMultinomial(prior)

    @classmethod
    def from_dim_alpha(cls, dim, alpha) -> "DirichletMultinomial":
        prior = DirichletPrior.from_dim_scale(dim, alpha)
        return DirichletMultinomial(prior)

    @classmethod
    def from_scale_mean(cls, scale, mean) -> "DirichletMultinomial":
        prior = DirichletPrior.from_scale_mean(scale, mean)
        return DirichletMultinomial(prior)

    def add_posterior_estimate(self) -> None:
        self.posteriors.append(self.get_posterior_dist())

    def summarize_posterior_estimate(self, lb: float = 2.5, ub: float = 97.5):
        mean_est = np.mean(self.posteriors, axis=0)
        ci_est = np.percentile(self.posteriors, [lb, ub], axis=0)
        return mean_est.tolist(), ci_est.tolist()

    def increment(self, observation: int) -> None:
        self.data.increment(observation)

    def decrement(self, observation: int) -> None:
        self.data.decrement(observation)

    def get_posterior_count(self, observation: int) -> float:
        return self.prior.vals[observation] + self.data.get_count(observation)

    def get_posterior_parameter(self) -> List[float]:
        return [
            self.get_posterior_count(observation)
            for observation in range(self.prior.dimension)
        ]

    def get_posterior_count_sum(self) -> float:
        return self.prior.scale + self.data.get_count_sum()

    def get_posterior_prob(self, observation: int) -> float:
        return self.get_posterior_count(observation) / self.get_posterior_count_sum()

    def get_posterior_dist(self) -> List[float]:
        return [
            self.get_posterior_prob(observation)
            for observation in range(self.prior.dimension)
        ]

    def sample_from_posterior(self) -> List[float]:
        return np.random.dirichlet(self.get_posterior_parameter()).tolist()

    def get_log_likelihood(self) -> float:
        llh = loggamma(self.prior.scale)
        for i_dim in range(self.prior.dimension):
            prior_val = self.prior.vals[i_dim]
            llh -= loggamma(prior_val)
            llh += loggamma(prior_val + self.data.get_count(i_dim))
        llh -= loggamma(self.prior.scale + self.data.get_count_sum())
        return llh


class NormalInverseGammaPrior:
    __slots__ = ["mu", "sigma", "alpha", "beta"]

    def __init__(self, mu: float, sigma: float, alpha: float, beta: float) -> None:
        self.mu = mu
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta

    def __repr__(self):
        return (
            f"mu = {self.mu}. sigma = {self.sigma}. "
            f"alpha = {self.alpha}. beta = {self.beta}."
        )

    @classmethod
    def from_hyperparameter(
        cls, mu: float, sigma: float, alpha: float, beta: float
    ) -> "NormalInverseGammaPrior":
        prior = cls(mu, sigma, alpha, beta)
        return prior


class Normal:
    def __init__(self) -> None:
        self.observations: List[float] = []
        self.count = 0
        self.sum = 0.0
        self.sum_squared = 0.0

    def __repr__(self):
        return (
            f"count = {self.count}. sum = {self.sum}. "
            f"sum_squared = {self.sum_squared}"
        )

    def add_observation(self, observation: float) -> None:
        self.observations.append(observation)
        self.count += 1
        self.sum += observation
        self.sum_squared += observation * observation

    def remove_observation(self, observation: float) -> None:
        self.observations.remove(observation)
        self.count -= 1
        self.sum -= observation
        self.sum_squared -= observation * observation

    def get_count(self) -> int:
        return self.count

    def get_sum(self) -> float:
        return self.sum

    def get_sum_squared(self) -> float:
        return self.sum_squared


class NormalInverseGammaNormal:
    def __init__(
        self,
        prior: NormalInverseGammaPrior = None,  # MLE if prior is None
        data: Normal = None,
    ) -> None:
        self.prior: NormalInverseGammaPrior = prior
        self.data: Normal = data if data is not None else Normal()
        self.mean: float = 0.0
        self.variance: float = 0.0
        if data is not None:
            self.estimate_parameters()

    def __repr__(self):
        return (
            f"prior: {self.prior}. data: {self.data}. "
            f"mean: {self.mean}. variance: {self.variance}"
        )

    @classmethod
    def from_prior_hyperparameters(
        cls, mu, sigma, alpha, beta
    ) -> "NormalInverseGammaNormal":
        prior = NormalInverseGammaPrior.from_hyperparameter(mu, sigma, alpha, beta)
        return NormalInverseGammaNormal(prior)

    def add_observation(self, observation: float, estimate: bool = True) -> None:
        self.data.add_observation(observation)
        if estimate:
            self.estimate_parameters()

    def remove_observation(self, observation: float, estimate: bool = True) -> None:
        self.data.remove_observation(observation)
        if estimate:
            self.estimate_parameters()

    def estimate_parameters(self) -> None:
        # MLE
        self.mean = self.data.sum / self.data.count
        self.variance = (
            self.data.sum_squared - self.data.count * self.mean * self.mean
        ) / (self.data.count - 1)

    def get_posterior_log_prob(self, observation: float) -> float:
        return norm.logpdf(observation, self.mean, math.sqrt(self.variance))

    def get_posterior_prob(self, observation: float) -> float:
        return norm.pdf(observation, self.mean, math.sqrt(self.variance))

    def get_log_likelihood(self) -> float:
        return sum(self.get_posterior_log_prob(x) for x in self.data.observations)
