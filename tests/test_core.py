import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from towerrec.baselines import EASE, ItemKNN, Popularity, top_k_excluding_seen
from towerrec.data import encode, split_global, split_per_user
from towerrec.metrics import bootstrap_ci, paired_bootstrap_diff, per_user_metrics
from towerrec.model import (
    HistoryIndex,
    HistorySampler,
    TowerConfig,
    TwoTower,
    embed_items,
    embed_users,
    in_batch_softmax_loss,
    train_two_tower,
    training_examples,
)


@pytest.fixture
def small():
    rng = np.random.default_rng(0)
    rows = []
    ts = 0
    # Two taste groups: even users like items 0-4, odd users like items 5-9.
    for user in range(1, 21):
        liked = range(1, 6) if user % 2 else range(6, 11)
        for item in rng.permutation(list(liked)):
            ts += 1
            rows.append((user, int(item), 5, ts))
    ratings = pd.DataFrame(rows, columns=["user", "item", "rating", "ts"])
    users = pd.DataFrame({"user": range(1, 21), "gender": ["F", "M"] * 10, "age": [25] * 20, "occupation": [1] * 20, "zip": ["1"] * 20})
    movies = pd.DataFrame(
        {"item": range(1, 11), "title": [f"Film {i} (199{i % 10})" for i in range(1, 11)], "genres": ["Action|Comedy"] * 5 + ["Drama"] * 5}
    )
    return ratings, users, movies


def test_global_split_has_no_time_overlap(small):
    ratings, _, _ = small
    split = split_global(ratings)
    assert split[split.fold == "train"].ts.max() < split[split.fold == "val"].ts.min()
    assert split[split.fold == "val"].ts.max() < split[split.fold == "test"].ts.min()


def test_global_split_keeps_equal_timestamps_in_one_fold():
    ratings = pd.DataFrame({"user": range(100), "item": range(100), "rating": 5, "ts": [1] * 85 + [2] * 15})
    split = split_global(ratings)
    assert split.groupby("ts").fold.nunique().max() == 1


def test_per_user_split_holds_out_each_users_latest(small):
    ratings, _, _ = small
    split = split_per_user(ratings, val=0.2, test=0.2)
    for _, group in split.groupby("user"):
        assert group[group.fold == "test"].ts.min() > group[group.fold == "train"].ts.max()


def test_encode_excludes_items_first_seen_in_test():
    ratings = pd.DataFrame({"user": [1, 1, 1], "item": [1, 1, 2], "rating": [5, 5, 5], "ts": [1, 2, 3]})
    ratings["fold"] = ["train", "val", "test"]
    users = pd.DataFrame({"user": [1], "gender": ["F"], "age": [25], "occupation": [1], "zip": ["1"]})
    movies = pd.DataFrame({"item": [1, 2], "title": ["A (1999)", "B (1999)"], "genres": ["Drama", "Drama"]})
    data = encode(ratings, users, movies)
    assert data.n_items == 1
    assert data.dropped_test_rows == 1


def test_truth_removes_seen_items(small):
    ratings, users, movies = small
    split = split_per_user(ratings, val=0.2, test=0.2)
    data = encode(split, users, movies)
    seen = data.matrix(("train", "val"))
    for user, items in data.truth("test", seen).items():
        assert all(not seen[user, item] for item in items)


def test_per_user_metrics_known_values():
    topk = np.array([[1, 2, 3], [7, 8, 9]])
    users = np.array([0, 1])
    truth = {0: {2}, 1: {5}}
    out = per_user_metrics(topk, users, truth, 3)
    assert out["recall@3"].tolist() == [1.0, 0.0]
    assert out["mrr@3"].tolist() == [0.5, 0.0]
    # One relevant item at rank 2: DCG = 1/log2(3), ideal DCG = 1/log2(2) = 1.
    assert out["ndcg@3"][0] == pytest.approx(1 / np.log2(3))


def test_ndcg_is_one_for_a_perfect_ranking():
    out = per_user_metrics(np.array([[4, 5, 6]]), np.array([0]), {0: {4, 5}}, 3)
    assert out["ndcg@3"][0] == pytest.approx(1.0)


def test_bootstrap_interval_contains_the_mean():
    values = np.random.default_rng(0).normal(size=500)
    low, high = bootstrap_ci(values)
    assert low < values.mean() < high


def test_paired_difference_detects_a_real_gap():
    rng = np.random.default_rng(0)
    b = rng.random(500)
    assert paired_bootstrap_diff(b + 0.1, b)["low"] > 0.09


def test_top_k_never_returns_seen_items():
    scores = np.array([[9.0, 8.0, 7.0, 6.0]])
    seen = sp.csr_matrix(np.array([[1, 0, 1, 0]], dtype=np.float32))
    assert top_k_excluding_seen(scores, seen, 2).tolist() == [[1, 3]]


@pytest.fixture
def interactions():
    # Users 0, 1 like items 0, 1, 2; users 2, 3 like items 3, 4, 5.
    x = np.zeros((4, 6), dtype=np.float32)
    x[0, [0, 1, 2]] = x[1, [0, 1, 2]] = 1
    x[2, [3, 4, 5]] = x[3, [3, 4, 5]] = 1
    return sp.csr_matrix(x)


@pytest.mark.parametrize("model", [EASE(l2=1.0), ItemKNN(neighbours=3, shrink=0.0)])
def test_item_models_recommend_from_the_same_taste_group(model, interactions):
    model.fit(interactions)
    history = sp.csr_matrix(np.array([[1, 1, 0, 0, 0, 0]], dtype=np.float32))
    assert top_k_excluding_seen(model.score(history), history, 1).tolist() == [[2]]


def test_ease_has_zero_diagonal(interactions):
    assert np.allclose(np.diag(EASE(l2=1.0).fit(interactions).weights), 0.0)


def test_cold_user_gets_popularity_order():
    x = sp.csr_matrix(np.array([[1, 1, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0]], dtype=np.float32))
    cold = sp.csr_matrix((1, 3), dtype=np.float32)
    for model in (EASE(l2=1.0), ItemKNN(neighbours=2), Popularity()):
        assert model.fit(x).score(cold).argmax() == 0


def test_training_history_only_contains_earlier_items():
    positives = pd.DataFrame({"user_idx": [0, 0, 0], "item_idx": [5, 6, 7], "ts": [1, 2, 3]})
    users, items, history = training_examples(positives, history_len=2, pad=99)
    assert items.tolist() == [5, 6, 7]
    assert history.tolist() == [[99, 99], [99, 5], [5, 6]]
    assert users.tolist() == [0, 0, 0]


def test_sampled_history_only_holds_the_users_other_items():
    users = np.array([0, 0, 0, 1, 1])
    items = np.array([5, 6, 7, 8, 9])
    sampler = HistorySampler(users, items, n_users=3, pad=99)
    rng = np.random.default_rng(0)
    batch_users = np.repeat(users, 200)
    batch_items = np.repeat(items, 200)
    drawn = sampler.sample(batch_users, batch_items, 4, rng)
    assert ((drawn != batch_items[:, None]) | (drawn == 99)).all()
    owned = {0: {5, 6, 7, 99}, 1: {8, 9, 99}}
    for user, row in zip(batch_users, drawn):
        assert set(row.tolist()) <= owned[int(user)]
    # Both a short and a full-length history should occur.
    counts = (drawn != 99).sum(axis=1)
    assert counts.min() <= 1 and counts.max() >= 2


def test_sampled_history_handles_a_single_item_user():
    sampler = HistorySampler(np.array([0]), np.array([5]), n_users=1, pad=99)
    drawn = sampler.sample(np.array([0]), np.array([5]), 3, np.random.default_rng(0))
    assert drawn.tolist() == [[99, 99, 99]]


def test_history_index_returns_latest_and_pads():
    positives = pd.DataFrame({"user_idx": [0, 0, 0], "item_idx": [5, 6, 7], "ts": [1, 2, 3]})
    index = HistoryIndex(positives, n_users=2)
    assert index.latest(np.array([0, 1]), 2, 99).tolist() == [[6, 7], [99, 99]]


def test_loss_masks_repeated_items_in_a_batch():
    # Rows 0 and 1 share an item, so neither should treat the other's copy as a negative.
    users = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), dim=-1)
    items = users.clone()
    ids = torch.tensor([3, 3, 8])
    masked = in_batch_softmax_loss(users, items, ids, temperature=0.1)
    assert masked.item() < 0.01


def test_log_q_correction_shrinks_the_logit_of_popular_items():
    # Both rows score both items equally. Row 0's target is rare, row 1's is popular. With the correction,
    # the popular item loses logit mass, so the rare target gets easier and the popular target harder.
    users = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    items = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    ids = torch.tensor([0, 1])
    log_q = torch.log(torch.tensor([0.01, 0.99]))
    logits = (users @ items.T - log_q[ids].unsqueeze(0)).detach()
    assert logits[0, 0] > logits[0, 1]
    assert in_batch_softmax_loss(users, items, ids, 1.0, log_q) != in_batch_softmax_loss(users, items, ids, 1.0)


def test_two_tower_training_reduces_the_loss(small):
    ratings, users, movies = small
    data = encode(split_per_user(ratings, val=0.2, test=0.2), users, movies)
    cfg = TowerConfig(dim=16, hidden=32, batch_size=8, epochs=40, patience=40, temperature=0.2, id_dropout=0.0, lr=5e-3, log_q=False)
    _, info = train_two_tower(data, cfg, validate=lambda m: 0.0, log=lambda _: None)
    assert info["curve"][-1]["loss"] < info["curve"][0]["loss"]


def test_two_tower_separates_taste_groups(small):
    ratings, users, movies = small
    data = encode(split_per_user(ratings, val=0.2, test=0.2), users, movies)
    cfg = TowerConfig(
        dim=16, hidden=32, batch_size=8, epochs=60, patience=60, temperature=0.2, id_dropout=0.0, lr=5e-3, log_q=False, use_features=False
    )
    # validate() increases every epoch, so the final weights are the ones kept.
    counter = iter(range(1000))
    model, _ = train_two_tower(data, cfg, validate=lambda m: float(next(counter)), log=lambda _: None)
    history = HistoryIndex(data.interactions(("train",), positive_only=True), data.n_users)
    items = embed_items(model)
    scores = embed_users(model, np.arange(data.n_users), history) @ items.T
    group_of_item = (data.item_ids > 5).astype(int)
    group_of_user = (data.user_ids % 2 == 0).astype(int)
    preferred = scores[:, group_of_item == 1].mean(axis=1) > scores[:, group_of_item == 0].mean(axis=1)
    assert (preferred.astype(int) == group_of_user).mean() == 1.0


def test_inner_product_of_exported_vectors_equals_the_training_score(small):
    ratings, users, movies = small
    data = encode(split_per_user(ratings, val=0.2, test=0.2), users, movies)
    cfg = TowerConfig(dim=8, hidden=16, temperature=0.2)
    model = TwoTower(data, cfg).eval()
    with torch.no_grad():
        model.item_bias.weight[:] = torch.arange(model.item_bias.num_embeddings, dtype=torch.float32).unsqueeze(-1)
        pad = model.item_id.padding_idx
        history = HistoryIndex(data.interactions(("train",), positive_only=True), data.n_users)
        user_ids = torch.tensor([0])
        hist = torch.as_tensor(history.latest(np.array([0]), cfg.history_len, pad))
        items = torch.arange(data.n_items)
        expected = (model.user_vector(user_ids, hist) @ model.item_vector(items).T) / cfg.temperature + model.bias(items)
    exported = embed_users(model, np.array([0]), history) @ embed_items(model).T
    assert np.allclose(exported, expected.numpy(), atol=1e-5)


def test_item_bias_is_learned_toward_popularity():
    # One very popular item among rare ones. With the logQ correction, the bias should rank it first.
    rows = [(user, 1, 5, user) for user in range(1, 41)]
    rows += [(user, 2 + user % 4, 5, 100 + user) for user in range(1, 41)]
    ratings = pd.DataFrame(rows, columns=["user", "item", "rating", "ts"])
    users = pd.DataFrame({"user": range(1, 41), "gender": ["F", "M"] * 20, "age": [25] * 40, "occupation": [1] * 40, "zip": ["1"] * 40})
    movies = pd.DataFrame({"item": range(1, 6), "title": [f"Film {i} (1995)" for i in range(1, 6)], "genres": ["Drama"] * 5})
    split = ratings.assign(fold="train")
    data = encode(split, users, movies)
    cfg = TowerConfig(
        dim=8, hidden=16, batch_size=16, epochs=50, patience=50, id_dropout=0.0, lr=0.02, use_history=False, use_features=False
    )
    counter = iter(range(1000))
    model, _ = train_two_tower(data, cfg, validate=lambda m: float(next(counter)), log=lambda _: None)
    with torch.no_grad():
        bias = model.bias(torch.arange(data.n_items)).numpy()
    assert bias.argmax() == 0


def test_towers_produce_unit_vectors_and_handle_cold_users(small):
    ratings, users, movies = small
    data = encode(split_per_user(ratings, val=0.2, test=0.2), users, movies)
    model = TwoTower(data, TowerConfig(dim=8, hidden=16, use_item_bias=False))
    history = HistoryIndex(data.interactions(("train",), positive_only=True), data.n_users)
    assert np.allclose(np.linalg.norm(embed_items(model), axis=1), 1.0, atol=1e-5)
    # A user with an empty history and an out-of-vocabulary ID still gets a valid vector.
    empty = HistoryIndex(data.interactions(("train",), positive_only=True).iloc[:0], data.n_users)
    model.user_known[:] = False
    vectors = embed_users(model, np.array([0, 1]), empty)
    assert np.isfinite(vectors).all()
    assert embed_users(model, np.array([0]), history).shape == (1, 8)
