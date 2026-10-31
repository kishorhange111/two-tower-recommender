"""Baselines the two-tower model has to beat.

Linear item-item models are not a formality: Dacrema et al. (RecSys 2019) found that most neural
recommenders they could reproduce lost to them. All baselines take a user's history as input, so a new
interaction changes the recommendation immediately without retraining.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def top_k_excluding_seen(scores: np.ndarray, seen: sp.csr_matrix, k: int) -> np.ndarray:
    """Indices of the k best unseen items per row, best first. ``scores`` is modified in place."""
    for row in range(scores.shape[0]):
        scores[row, seen.indices[seen.indptr[row] : seen.indptr[row + 1]]] = -np.inf
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1)
    return np.take_along_axis(part, order, axis=1)


class Popularity:
    """Most-liked items in the training fold, the same list for everyone."""

    name = "popularity"

    def fit(self, train: sp.csr_matrix) -> Popularity:
        counts = np.asarray(train.sum(axis=0)).ravel()
        self.scores = counts / max(counts.max(), 1.0)
        return self

    def score(self, history: sp.csr_matrix) -> np.ndarray:
        return np.tile(self.scores, (history.shape[0], 1))


class ItemKNN:
    """Cosine item-item neighbours with a shrinkage term that distrusts small co-occurrence counts."""

    name = "item_knn"

    def __init__(self, neighbours: int = 100, shrink: float = 10.0):
        self.neighbours = neighbours
        self.shrink = shrink

    def fit(self, train: sp.csr_matrix) -> ItemKNN:
        co = (train.T @ train).toarray().astype(np.float32)
        norm = np.sqrt(np.diag(co))
        sim = co / (np.outer(norm, norm) + self.shrink)
        np.fill_diagonal(sim, 0.0)
        if self.neighbours < sim.shape[0]:
            weakest = np.argpartition(-sim, self.neighbours, axis=1)[:, self.neighbours :]
            np.put_along_axis(sim, weakest, 0.0, axis=1)
        self.sim = sim
        self.popularity = Popularity().fit(train).scores
        return self

    def score(self, history: sp.csr_matrix) -> np.ndarray:
        # The tiny popularity term only breaks ties, so a user with no history gets the popularity list.
        return np.asarray(history @ self.sim) + 1e-6 * self.popularity


class EASE:
    """EASE^R (Steck, WWW 2019): closed-form item-item regression with a zero diagonal."""

    name = "ease"

    def __init__(self, l2: float = 500.0):
        self.l2 = l2

    def fit(self, train: sp.csr_matrix) -> EASE:
        gram = (train.T @ train).toarray().astype(np.float64)
        gram[np.diag_indices_from(gram)] += self.l2
        precision = np.linalg.inv(gram)
        weights = precision / (-np.diag(precision))
        np.fill_diagonal(weights, 0.0)
        self.weights = weights.astype(np.float32)
        self.popularity = Popularity().fit(train).scores
        return self

    def score(self, history: sp.csr_matrix) -> np.ndarray:
        return np.asarray(history @ self.weights) + 1e-6 * self.popularity
