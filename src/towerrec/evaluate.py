"""Full evaluation protocol.

Everything that has weights is fit on the train fold only. Hyper-parameters and early stopping use val. At
test time, models read each user's history up to the end of val, as a deployed model with a stale training
run but a fresh activity log would. Warm and cold users are reported apart, because averaging them hides
what each model is actually doing.

    towerrec-eval --split global --out results/global
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from .baselines import EASE, ItemKNN, Popularity, top_k_excluding_seen
from .data import Dataset, encode, load_movielens, make_split
from .metrics import paired_bootstrap_diff, per_user_metrics, summarise
from .model import HistoryIndex, TowerConfig, TwoTower, embed_items, embed_users, train_two_tower
from .ranker import FEATURES, FeatureBuilder, rerank, train_ranker
from .serving import export

K = 100
BATCH = 512

ABLATION = {
    "id_only": TowerConfig(use_user_id=True, use_history=False, use_features=False),
    "id_features": TowerConfig(use_user_id=True, use_history=False, use_features=True),
    "history_only": TowerConfig(use_features=False),
    "full_with_user_id": TowerConfig(use_user_id=True),
    "full_ordered_history": TowerConfig(history_mode="before"),
    "full_no_item_bias": TowerConfig(use_item_bias=False),
    "full_no_logq": TowerConfig(log_q=False),
    "full": TowerConfig(),
}


def baseline_topk(model, users: np.ndarray, history: sp.csr_matrix, seen: sp.csr_matrix, k: int = K) -> np.ndarray:
    parts = []
    for start in range(0, len(users), BATCH):
        chunk = users[start : start + BATCH]
        parts.append(top_k_excluding_seen(model.score(history[chunk]), seen[chunk], k))
    return np.concatenate(parts) if parts else np.empty((0, k), dtype=np.int64)


def tower_topk(user_vec: np.ndarray, item_vec: np.ndarray, users: np.ndarray, seen: sp.csr_matrix, k: int = K) -> np.ndarray:
    parts = []
    for start in range(0, len(users), BATCH):
        chunk = users[start : start + BATCH]
        scores = user_vec[start : start + BATCH] @ item_vec.T
        parts.append(top_k_excluding_seen(scores, seen[chunk], k))
    return np.concatenate(parts) if parts else np.empty((0, k), dtype=np.int64)


def split_warm_cold(users: np.ndarray, history: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray(history.sum(axis=1)).ravel()[users]
    return users[counts > 0], users[counts == 0]


class Protocol:
    """Holds one split's matrices and answers 'how good is this top-k?' for every model the same way."""

    def __init__(self, data: Dataset):
        self.data = data
        self.train_pos = data.matrix(("train",), positive_only=True)
        self.past_pos = data.matrix(("train", "val"), positive_only=True)
        self.seen_for_val = data.matrix(("train",))
        self.seen_for_test = data.matrix(("train", "val"))
        self.val_truth = data.truth("val", self.seen_for_val)
        self.test_truth = data.truth("test", self.seen_for_test)
        self.val_users = np.array(sorted(self.val_truth), dtype=np.int64)
        self.test_users = np.array(sorted(self.test_truth), dtype=np.int64)
        self.warm, self.cold = split_warm_cold(self.test_users, self.past_pos)
        # 100 means the most-liked item in the catalogue. It shows whether a model is chasing or avoiding popular items.
        likes = np.asarray(self.train_pos.sum(axis=0)).ravel()
        self.popularity_percentile = 100.0 * likes.argsort().argsort() / max(data.n_items - 1, 1)
        self.val_history = HistoryIndex(data.interactions(("train",), positive_only=True), data.n_users)
        self.test_history = HistoryIndex(data.interactions(("train", "val"), positive_only=True), data.n_users)

    def report(self, topk: np.ndarray, users: np.ndarray) -> dict:
        lookup = {int(u): i for i, u in enumerate(users)}

        def subset(group: np.ndarray) -> dict:
            rows = [lookup[int(u)] for u in group]
            return summarise(topk[rows], group, self.test_truth, self.data.n_items)

        overall = summarise(topk, users, self.test_truth, self.data.n_items)
        overall["popularity_percentile@10"] = float(self.popularity_percentile[topk[:, :10]].mean())
        return {"all": overall, "warm": subset(self.warm), "cold": subset(self.cold)}

    def val_score(self, topk: np.ndarray) -> float:
        return float(per_user_metrics(topk, self.val_users, self.val_truth, 50)["recall@50"].mean())


def select_baseline(protocol: Protocol, factory, grid: list[dict], log) -> tuple[object, dict]:
    best, best_score, best_params = None, -1.0, None
    for params in grid:
        model = factory(**params).fit(protocol.train_pos)
        score = protocol.val_score(baseline_topk(model, protocol.val_users, protocol.train_pos, protocol.seen_for_val))
        log(f"  {factory.name} {params} val recall@50 {score:.4f}")
        if score > best_score:
            best, best_score, best_params = model, score, params
    return best, {"params": best_params, "val_recall@50": best_score}


def run_tower(protocol: Protocol, cfg: TowerConfig, log) -> tuple[TwoTower, dict]:
    def validate(model: TwoTower) -> float:
        items = embed_items(model)
        users = embed_users(model, protocol.val_users, protocol.val_history)
        return protocol.val_score(tower_topk(users, items, protocol.val_users, protocol.seen_for_val))

    start = time.perf_counter()
    model, info = train_two_tower(protocol.data, cfg, validate, log)
    info["train_seconds"] = time.perf_counter() - start
    return model, info


def test_topk(protocol: Protocol, model: TwoTower) -> np.ndarray:
    items = embed_items(model)
    users = embed_users(model, protocol.test_users, protocol.test_history)
    return tower_topk(users, items, protocol.test_users, protocol.seen_for_test)


def run_reranker(protocol: Protocol, tower: TwoTower, ease: EASE, log, n_candidates: int = K) -> tuple[np.ndarray, dict]:
    """Fit on val labels with train-only features, apply to test with train+val features."""
    item_vec = embed_items(tower)

    def stage(users, history_index, history_matrix, seen, folds):
        user_vec = embed_users(tower, users, history_index)
        candidates = tower_topk(user_vec, item_vec, users, seen, n_candidates)
        tower_scores = np.einsum("ij,ikj->ik", user_vec, item_vec[candidates])
        ease_parts = []
        for start in range(0, len(users), BATCH):
            chunk = users[start : start + BATCH]
            ease_parts.append(np.take_along_axis(ease.score(history_matrix[chunk]), candidates[start : start + BATCH], axis=1))
        features = FeatureBuilder(protocol.data, folds).build(users, candidates, tower_scores, np.concatenate(ease_parts))
        return candidates, features

    candidates, features = stage(protocol.val_users, protocol.val_history, protocol.train_pos, protocol.seen_for_val, ("train",))
    labels = np.concatenate(
        [np.isin(candidates[r], list(protocol.val_truth[int(u)])).astype(int) for r, u in enumerate(protocol.val_users)]
    )
    start = time.perf_counter()
    booster = train_ranker(features, labels, n_candidates)
    info = {
        "train_seconds": time.perf_counter() - start,
        "best_iteration": booster.best_iteration,
        "val_users": len(protocol.val_users),
        "importance_gain": dict(zip(FEATURES, booster.feature_importance("gain").round(1).tolist())),
    }
    log(f"  ranker stopped at {booster.best_iteration} trees")
    candidates, features = stage(protocol.test_users, protocol.test_history, protocol.past_pos, protocol.seen_for_test, ("train", "val"))
    return rerank(booster, features, candidates), info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["global", "per_user"], default="global")
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="results/global")
    parser.add_argument("--seeds", type=int, default=3, help="seeds for the final tower configuration")
    parser.add_argument("--configs", nargs="*", default=list(ABLATION))
    parser.add_argument("--temperatures", nargs="*", type=float, default=[0.1, 0.2])
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        print(message, flush=True)

    ratings, users, movies = load_movielens(args.data)
    data = encode(make_split(ratings, args.split), users, movies)
    protocol = Protocol(data)
    folds = data.frame["fold"].value_counts().to_dict()
    summary: dict = {
        "split": args.split,
        "folds": {k: int(v) for k, v in folds.items()},
        "n_users": data.n_users,
        "n_catalogue_items": data.n_items,
        "test_rows_on_items_outside_catalogue": data.dropped_test_rows,
        "test_users": {"all": len(protocol.test_users), "warm": len(protocol.warm), "cold": len(protocol.cold)},
        "models": {},
    }
    log(json.dumps({k: summary[k] for k in ("folds", "n_catalogue_items", "test_users")}))

    topks: dict[str, np.ndarray] = {}

    log("popularity")
    popularity = Popularity().fit(protocol.train_pos)
    topks["popularity"] = baseline_topk(popularity, protocol.test_users, protocol.past_pos, protocol.seen_for_test)
    summary["models"]["popularity"] = protocol.report(topks["popularity"], protocol.test_users)

    log("item_knn")
    grid = [{"neighbours": n, "shrink": s} for n in (50, 100, 200) for s in (0.0, 10.0, 50.0)]
    knn, info = select_baseline(protocol, ItemKNN, grid, log)
    topks["item_knn"] = baseline_topk(knn, protocol.test_users, protocol.past_pos, protocol.seen_for_test)
    summary["models"]["item_knn"] = {**protocol.report(topks["item_knn"], protocol.test_users), "selection": info}

    log("ease")
    ease, info = select_baseline(protocol, EASE, [{"l2": v} for v in (100.0, 300.0, 500.0, 1000.0, 3000.0)], log)
    topks["ease"] = baseline_topk(ease, protocol.test_users, protocol.past_pos, protocol.seen_for_test)
    summary["models"]["ease"] = {**protocol.report(topks["ease"], protocol.test_users), "selection": info}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # Temperature is tuned once on the full configuration, then shared by the whole ablation.
    best_temperature, best_val, best_model, best_info = None, -1.0, None, None
    for temperature in args.temperatures:
        log(f"tower full, temperature {temperature}")
        cfg = replace(ABLATION["full"], temperature=temperature)
        model, info = run_tower(protocol, cfg, log)
        if info["best_val"] > best_val:
            best_temperature, best_val, best_model, best_info = temperature, info["best_val"], model, info
    summary["temperature"] = {"selected": best_temperature, "val_recall@50": best_val, "grid": args.temperatures}

    for name in args.configs:
        log(f"tower {name}")
        if name == "full":
            model, info = best_model, best_info
        else:
            model, info = run_tower(protocol, replace(ABLATION[name], temperature=best_temperature), log)
        key = f"tower_{name}"
        topks[key] = test_topk(protocol, model)
        summary["models"][key] = {**protocol.report(topks[key], protocol.test_users), "training": info}
        (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # Seed variance is reported apart from the user bootstrap: they answer different questions.
    seeds = {0: summary["models"]["tower_full"]["all"]}
    for seed in range(1, args.seeds):
        log(f"tower full, seed {seed}")
        model, _ = run_tower(protocol, replace(ABLATION["full"], temperature=best_temperature, seed=seed), log)
        seeds[seed] = protocol.report(test_topk(protocol, model), protocol.test_users)["all"]
    summary["tower_full_seeds"] = {
        metric: {
            "mean": float(np.mean([s[metric]["mean"] for s in seeds.values()])),
            "std": float(np.std([s[metric]["mean"] for s in seeds.values()])),
        }
        for metric in ("recall@10", "recall@50", "ndcg@10")
    }

    log("reranker")
    topks["tower_full_reranked"], info = run_reranker(protocol, best_model, ease, log)
    summary["models"]["tower_full_reranked"] = {**protocol.report(topks["tower_full_reranked"], protocol.test_users), "training": info}

    def user_metric(name: str, metric: str, k: int) -> np.ndarray:
        return per_user_metrics(topks[name], protocol.test_users, protocol.test_truth, k)[f"{metric}@{k}"]

    comparisons = [
        ("ndcg", 10, "tower_full_reranked", "tower_full"),
        ("ndcg", 10, "tower_full_reranked", "ease"),
        ("ndcg", 10, "tower_full", "ease"),
        ("recall", 100, "tower_full", "ease"),
        ("recall", 100, "tower_full", "tower_full_no_logq"),
    ]
    # A partial --configs run has nothing to compare against for some of these, so they are skipped.
    summary["paired_differences"] = {
        f"{metric}@{k} {a} - {b}": paired_bootstrap_diff(user_metric(a, metric, k), user_metric(b, metric, k))
        for metric, k, a, b in comparisons
        if a in topks and b in topks
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    export(best_model, data, ("train", "val"), out / "artifacts")
    log("done")


if __name__ == "__main__":
    main()
