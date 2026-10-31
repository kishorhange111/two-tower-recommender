"""Second stage: LightGBM LambdaRank over the retriever's candidates.

The ranker's labels come from the validation fold, which the retriever never trained on. Training it on
the retriever's own training data would teach it to trust scores that are optimistic on exactly those pairs.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import scipy.sparse as sp

from .data import Dataset

FEATURES = [
    "tower_score", "tower_rank", "ease_score", "log_item_pop", "item_mean_rating", "item_year",
    "genre_affinity", "log_user_history", "user_mean_rating", "gender", "age",
]  # fmt: skip


class FeatureBuilder:
    """Precomputes statistics from the given (past) folds so features never use future ratings."""

    def __init__(self, data: Dataset, folds: tuple[str, ...]):
        self.data = data
        rows = data.interactions(folds)
        liked = data.interactions(folds, positive_only=True)
        self.item_pop = np.log1p(np.bincount(liked["item_idx"], minlength=data.n_items))
        global_mean = rows["rating"].mean()
        # Shrink toward the global mean so an item with two 5-star ratings does not look best.
        sums = np.bincount(rows["item_idx"], weights=rows["rating"], minlength=data.n_items)
        counts = np.bincount(rows["item_idx"], minlength=data.n_items)
        self.item_mean = (sums + 10 * global_mean) / (counts + 10)
        user_sums = np.bincount(rows["user_idx"], weights=rows["rating"], minlength=data.n_users)
        user_counts = np.bincount(rows["user_idx"], minlength=data.n_users)
        self.user_mean = (user_sums + 10 * global_mean) / (user_counts + 10)
        self.user_history = np.log1p(user_counts)
        liked_matrix = sp.csr_matrix(
            (np.ones(len(liked), dtype=np.float32), (liked["user_idx"], liked["item_idx"])), shape=(data.n_users, data.n_items)
        )
        profile = np.asarray(liked_matrix @ data.item_genres)
        self.genre_profile = profile / np.maximum(np.linalg.norm(profile, axis=1, keepdims=True), 1e-9)
        norm = np.maximum(np.linalg.norm(data.item_genres, axis=1, keepdims=True), 1e-9)
        self.item_genre_unit = data.item_genres / norm

    def build(self, users: np.ndarray, candidates: np.ndarray, tower_scores: np.ndarray, ease_scores: np.ndarray) -> np.ndarray:
        """Feature matrix of shape (len(users) * n_candidates, n_features)."""
        n_users, n_cand = candidates.shape
        flat_items = candidates.ravel()
        repeated = np.repeat(users, n_cand)
        rank = np.tile(np.arange(n_cand), n_users)
        affinity = (self.genre_profile[repeated] * self.item_genre_unit[flat_items]).sum(axis=1)
        return np.column_stack(
            [
                tower_scores.ravel(),
                rank,
                ease_scores.ravel(),
                self.item_pop[flat_items],
                self.item_mean[flat_items],
                self.data.item_year[flat_items],
                affinity,
                self.user_history[repeated],
                self.user_mean[repeated],
                self.data.gender[repeated],
                self.data.age[repeated],
            ]
        ).astype(np.float32)


def train_ranker(features: np.ndarray, labels: np.ndarray, n_candidates: int, seed: int = 0) -> lgb.Booster:
    """Fit LambdaRank with early stopping on a 20 % hold-out of users. Rows are grouped per user."""
    n_users = len(labels) // n_candidates
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_users)
    cut = int(0.8 * n_users)

    def rows(user_rows: np.ndarray) -> np.ndarray:
        return (user_rows[:, None] * n_candidates + np.arange(n_candidates)).ravel()

    fit, stop = rows(order[:cut]), rows(order[cut:])
    train_set = lgb.Dataset(features[fit], labels[fit], group=np.full(cut, n_candidates), feature_name=FEATURES, free_raw_data=False)
    stop_set = lgb.Dataset(
        features[stop], labels[stop], group=np.full(n_users - cut, n_candidates), reference=train_set, free_raw_data=False
    )
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "seed": seed,
        "verbosity": -1,
    }
    return lgb.train(params, train_set, num_boost_round=500, valid_sets=[stop_set], callbacks=[lgb.early_stopping(30, verbose=False)])


def rerank(model: lgb.Booster, features: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    scores = model.predict(features, num_iteration=model.best_iteration).reshape(candidates.shape)
    order = np.argsort(-scores, axis=1)
    return np.take_along_axis(candidates, order, axis=1)
