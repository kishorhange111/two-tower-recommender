"""Top-K ranking metrics, always against the full catalogue.

Sampled-negative evaluation (rank the target among 100 random items) is deliberately not used: it can
reorder models relative to their true full-ranking performance (Krichene and Rendle, KDD 2020).
"""

from __future__ import annotations

import numpy as np


def per_user_metrics(topk: np.ndarray, users: np.ndarray, truth: dict[int, set[int]], k: int) -> dict[str, np.ndarray]:
    """Recall@k, NDCG@k and reciprocal rank (within k) for each row of ``topk``."""
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    ideal = np.cumsum(discounts)
    recall = np.zeros(len(users))
    ndcg = np.zeros(len(users))
    rr = np.zeros(len(users))
    for row, user in enumerate(users):
        relevant = truth[int(user)]
        hits = np.isin(topk[row, :k], list(relevant))
        count = hits.sum()
        if count == 0:
            continue
        recall[row] = count / len(relevant)
        ndcg[row] = (hits * discounts).sum() / ideal[min(len(relevant), k) - 1]
        rr[row] = 1.0 / (np.argmax(hits) + 1)
    return {f"recall@{k}": recall, f"ndcg@{k}": ndcg, f"mrr@{k}": rr}


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """Percentile 95 % interval of the mean, resampling users."""
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[samples].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def paired_bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict[str, float]:
    """Mean of ``a - b`` per user, with a 95 % interval. An interval excluding 0 is a real difference."""
    diff = a - b
    low, high = bootstrap_ci(diff, n_boot, seed)
    return {"diff": float(diff.mean()), "low": low, "high": high}


def summarise(topk: np.ndarray, users: np.ndarray, truth: dict[int, set[int]], n_items: int, ks: tuple[int, ...] = (10, 50, 100)) -> dict:
    """Aggregate metrics with confidence intervals, plus how much of the catalogue gets recommended."""
    result: dict = {"users": len(users)}
    if len(users) == 0:
        return result
    for k in ks:
        for name, values in per_user_metrics(topk, users, truth, k).items():
            low, high = bootstrap_ci(values)
            result[name] = {"mean": float(values.mean()), "low": low, "high": high}
    result["coverage@10"] = float(len(np.unique(topk[:, :10])) / n_items)
    return result
