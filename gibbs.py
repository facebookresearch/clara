#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.

import logging
from typing import List, Optional

import numpy as np

from base import BaseModel
from stats import DirichletMultinomial, DirichletPrior, NormalInverseGammaNormal


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClaraGibbs(BaseModel):
    def __init__(
        self,
        burn_in: int = 1000,
        num_samples: int = 1000,
        sample_lag: int = 1,
        theta_scale: Optional[float] = None,
        theta_mean: Optional[List[float]] = None,
        psi_scale: Optional[List[float]] = None,
        psi_mean: Optional[List[List[float]]] = None,
    ):
        super().__init__("ClaraGibbs")
        self.burn_in = burn_in
        self.num_samples = num_samples
        self.sample_lag = sample_lag
        self.theta_scale = theta_scale
        self.theta_mean = theta_mean
        self.psi_scale = psi_scale
        self.psi_mean = psi_mean

    def _increment(
        self,
        z: int,
        item_ratings: List[int],
        item_labelers: List[int],
        item_scores: Optional[List[List[float]]] = None,
    ):
        self.theta.increment(z)
        for j in range(len(item_ratings)):
            self.psi[item_labelers[j]][z].increment(item_ratings[j])
        if item_scores is not None:
            for c in range(self.C):
                self.phi[c][z].add_observation(item_scores[c][z], False)

    def _decrement(
        self,
        z: int,
        item_ratings: List[int],
        item_labelers: List[int],
        item_scores: Optional[List[List[float]]] = None,
    ):
        self.theta.decrement(z)
        for j in range(len(item_ratings)):
            self.psi[item_labelers[j]][z].decrement(item_ratings[j])
        if item_scores is not None:
            for c in range(self.C):
                self.phi[c][z].remove_observation(item_scores[c][z], False)

    def _sample(
        self,
        item_ratings: List[int],
        item_labelers: List[int],
        item_scores: Optional[List[List[float]]] = None,
    ):
        probs = np.array([self.theta.get_posterior_prob(r) for r in range(self.R)])
        for k in range(self.R):
            for j in range(len(item_ratings)):
                labeler = item_labelers[j]
                rating = item_ratings[j]
                probs[k] *= self.psi[labeler][k].get_posterior_prob(rating)
            if item_scores is not None:
                for c in range(self.C):
                    probs[k] *= self.phi[c][k].get_posterior_prob(item_scores[c][k])

        norm_probs = probs / np.sum(probs)
        return np.random.choice(self.R, p=norm_probs)

    def _update_gaussians(self):
        for c in range(self.C):
            for r in range(self.R):
                self.phi[c][r].estimate_parameters()

    def _get_log_likelihood(self) -> float:
        llh = 0.0
        llh += self.theta.get_log_likelihood()
        for l_psi in self.psi:
            for r_psi in l_psi:
                llh += r_psi.get_log_likelihood()
        if self.C > 0:
            for c_phi in self.phi:
                for r_phi in c_phi:
                    llh += r_phi.get_log_likelihood()
        return llh

    def _get_priors(
        self, ratings: np.array, labelers: np.array, scores: Optional[np.array] = None
    ):
        logger.info("Getting priors ...")

        # theta
        if self.theta_scale is None:
            self.theta_scale = 1.0
        logger.info(f"  theta_scale = {self.theta_scale}")

        if self.theta_mean is None:
            flatten_ratings = np.hstack(ratings)
            obs_ratings, obs_counts = np.unique(flatten_ratings, return_counts=True)

            theta_counts = np.zeros(self.R, dtype=float)
            for i in range(len(obs_ratings)):
                theta_counts[obs_ratings[i]] = 1.0 + obs_counts[i]
            self.theta_mean = theta_counts / np.sum(theta_counts)
        logger.info(f"  theta_mean = {self.theta_mean}")

        theta_prior = DirichletPrior.from_scale_mean(self.theta_scale, self.theta_mean)
        logger.info(f" theta_prior = {theta_prior}")

        # psi
        if self.psi_scale is None:
            self.psi_scale = [1.0] * self.R
        logger.info(f"  psi_scale = {self.psi_scale}")

        if self.psi_mean is None:
            diag_value = 0.75
            off_diag_value = (1.0 - diag_value) / (self.R - 1)
            self.psi_mean = np.zeros((self.R, self.R), dtype=float)
            for r in range(self.R):
                for o in range(self.R):
                    self.psi_mean[r][o] = diag_value if r == o else off_diag_value
        logger.info(f"  psi_mean = {self.psi_mean.tolist()}")

        psi_prior = [
            DirichletPrior.from_scale_mean(self.psi_scale[r], self.psi_mean[r])
            for r in range(self.R)
        ]
        logger.info(f" psi_prior = {psi_prior}")

        # phi
        phi_prior = None
        if scores is not None:
            phi_prior = [[None for r in range(self.R)] for c in range(self.C)]
        logger.info(f" phi_prior = {phi_prior}")

        return theta_prior, psi_prior, phi_prior

    def _init(
        self,
        ratings: np.array,
        labelers: np.array,
        true_ratings: np.array,
        scores: Optional[np.array] = None,
    ):
        logger.info("Initializing ...")
        N = len(ratings)

        #  process priors
        theta_prior, psi_prior, phi_prior = self._get_priors(ratings, labelers, scores)

        self.theta = DirichletMultinomial(prior=theta_prior)
        self.psi: List[List[DirichletMultinomial]] = [
            [DirichletMultinomial(psi_prior[r]) for r in range(self.R)]
            for j in range(self.A)
        ]
        if scores is not None:
            self.phi: List[List[NormalInverseGammaNormal]] = [
                [NormalInverseGammaNormal(phi_prior[c][r]) for r in range(self.R)]
                for c in range(self.C)
            ]

        # initialize assignments
        self.zs = np.empty(N, dtype=int)
        for n in range(N):
            if true_ratings[n] == -1:
                (values, counts) = np.unique(ratings[n], return_counts=True)
                z = values[np.argmax(counts)]
            else:
                z = true_ratings[n]
            self._increment(
                z=z,
                item_ratings=ratings[n],
                item_labelers=labelers[n],
                item_scores=None if scores is None else scores[n].tolist(),
            )
            self.zs[n] = z

        # update Gaussians
        if self.C > 0:
            self._update_gaussians()

        self._log_status()

    def _log_status(self):
        logger.info(f"  llh = {self._get_log_likelihood()}")
        logger.info(f"  theta = {self.theta}")
        for l in range(self.A):
            for r in range(self.R):
                logger.info(f"  psi[{l}][{r}] = {self.psi[l][r]}")
        if self.C != 0:
            for c in range(self.C):
                for r in range(self.R):
                    logger.info(f"  phi[{c}][{r}] = {self.phi[c][r]}")

    def _iterate(
        self,
        ratings: np.array,
        labelers: np.array,
        true_ratings: np.array,
        scores: Optional[np.array] = None,
    ):
        logger.info("Sampling ...")
        N = len(ratings)
        max_iters = self.burn_in + self.num_samples * self.sample_lag
        num_logs = 10
        log_step = (int)(max_iters / num_logs)

        indices = np.array(range(N))

        for iter in range(max_iters):
            n_changes = 0
            np.random.shuffle(indices)

            for n in indices:
                self._decrement(
                    z=self.zs[n],
                    item_ratings=ratings[n],
                    item_labelers=labelers[n],
                    item_scores=None if scores is None else scores[n].tolist(),
                )

                z = self._sample(
                    item_ratings=ratings[n],
                    item_labelers=labelers[n],
                    item_scores=None if scores is None else scores[n].tolist(),
                )
                if self.zs[n] != z:
                    n_changes += 1
                self.zs[n] = z

                self._increment(
                    z=self.zs[n],
                    item_ratings=ratings[n],
                    item_labelers=labelers[n],
                    item_scores=None if scores is None else scores[n].tolist(),
                )

            if self.C > 0:
                self._update_gaussians()

            # collect samples
            is_stored = iter >= self.burn_in and iter % self.sample_lag == 0
            if is_stored:
                self.theta.add_posterior_estimate()
                for l_psi in self.psi:
                    for r_psi in l_psi:
                        r_psi.add_posterior_estimate()

            if iter % log_step == 0:
                logger.info(f" Iter {iter} / {max_iters}")
                logger.info(f"  n_changes = {n_changes} / {N}")
                self._log_status()

        logger.info(f"Done sampling!")

    def fit(
        self,
        R: int,  # num. unique ratings
        A: int,  # num. unique labelers
        ratings: np.array,
        labelers: Optional[np.array] = None,
        scores: Optional[np.array] = None,  # (N x C x R)-shaped array
        true_ratings: Optional[np.array] = None,
    ):
        logger.info("Fitting ...")

        N = len(ratings)
        self.R = R
        self.A = A
        self.C = 0 if scores is None else scores.shape[1]
        logger.info(f" N = {N}")
        logger.info(f" R = {self.R}")
        logger.info(f" A = {self.A}")
        logger.info(f" C = {self.C}")

        # standardize observed variables
        if labelers is None:  # use a single confusion matrix if None
            labelers = np.array(
                [np.zeros(len(ratings[n]), dtype=int) for n in range(N)]
            )
        if true_ratings is None:
            true_ratings = np.repeat(-1, N)

        # initialize latent variables
        self._init(ratings, labelers, true_ratings, scores)

        # sample
        self._iterate(ratings, labelers, true_ratings, scores)

    def predict(self, **kwargs):
        pass

    def get_prevalence(self):
        mean_est, ci_est = self.theta.summarize_posterior_estimate()
        return {"mean": mean_est, "ci": ci_est}

    def get_confusion_matrix(self, labeler_id: int):
        labeler_psi = self.psi[labeler_id]
        estimates = []
        for r in range(self.R):
            mean_est, ci_est = labeler_psi[r].summarize_posterior_estimate()
            estimates.append({"mean": mean_est, "ci": ci_est})
        return estimates
