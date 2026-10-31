import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from towerrec.ann_benchmark import benchmark_size, scale_catalogue
from towerrec.data import encode, split_per_user
from towerrec.model import HistoryIndex, TowerConfig, TwoTower
from towerrec.serving import Recommender, create_app, export, load

FOLDS = ("train", "val", "test")


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    rows = []
    ts = 0
    for user in range(1, 21):
        for item in rng.permutation(list(range(1, 6) if user % 2 else range(6, 11))):
            ts += 1
            rows.append((user, int(item), 5, ts))
    ratings = pd.DataFrame(rows, columns=["user", "item", "rating", "ts"])
    users = pd.DataFrame({"user": range(1, 21), "gender": ["F", "M"] * 10, "age": [25] * 20, "occupation": [1] * 20, "zip": ["1"] * 20})
    movies = pd.DataFrame({"item": range(1, 11), "title": [f"Film {i} (1995)" for i in range(1, 11)], "genres": ["Drama"] * 10})
    return encode(split_per_user(ratings, val=0.2, test=0.2), users, movies)


@pytest.fixture
def recommender(data):
    torch_model = TwoTower(data, TowerConfig(dim=8, hidden=16))
    history = HistoryIndex(data.interactions(("train",), positive_only=True), data.n_users)
    return Recommender(torch_model, data, history, data.matrix(("train",)))


def test_recommendations_never_repeat_seen_items(recommender, data):
    user_id = int(data.user_ids[0])
    seen = recommender.seen[0].indices
    seen_ids = {int(data.item_ids[i]) for i in seen}
    items = [row["item_id"] for row in recommender.recommend(user_id, k=20)]
    assert not seen_ids & set(items)
    assert len(items) == data.n_items - len(seen_ids)


def test_recent_items_change_the_result_and_are_excluded(recommender, data):
    user_id = int(data.user_ids[0])
    unseen = [row["item_id"] for row in recommender.recommend(user_id, k=20)]
    updated = [row["item_id"] for row in recommender.recommend(user_id, k=20, recent_items=[unseen[0]])]
    assert unseen[0] not in updated


def test_unknown_recent_items_are_ignored(recommender, data):
    assert recommender.recommend(int(data.user_ids[0]), k=3, recent_items=[999999])


def test_scores_are_sorted(recommender, data):
    scores = [row["score"] for row in recommender.recommend(int(data.user_ids[0]), k=20)]
    assert scores == sorted(scores, reverse=True)


def test_unknown_user_raises(recommender):
    with pytest.raises(KeyError):
        recommender.recommend(424242)


def test_api(recommender, data):
    with TestClient(create_app(recommender)) as client:
        assert client.get("/health").json()["items"] == data.n_items
        body = client.post("/recommend", json={"user_id": int(data.user_ids[0]), "k": 2}).json()
        assert len(body["items"]) == 2
        assert {"item_id", "title", "score"} <= set(body["items"][0])
        assert client.post("/recommend", json={"user_id": 424242}).status_code == 404
        assert client.post("/recommend", json={"user_id": int(data.user_ids[0]), "k": 0}).status_code == 422


def test_export_and_load_give_identical_recommendations(recommender, data, tmp_path):
    export(recommender.model, data, FOLDS[:1], tmp_path)
    loaded = load(tmp_path)
    user_id = int(data.user_ids[0])
    assert loaded.recommend(user_id, k=3) == recommender.recommend(user_id, k=3)


def test_scale_catalogue_keeps_originals_first():
    vectors = np.random.default_rng(0).normal(size=(10, 4)).astype(np.float32)
    scaled = scale_catalogue(vectors, 35)
    assert scaled.shape == (35, 4)
    assert np.array_equal(scaled[:10], vectors)
    assert not np.array_equal(scaled[10:20], vectors)
    assert scale_catalogue(vectors, 5).shape == (5, 4)


def test_exact_search_has_perfect_recall_in_the_benchmark():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(2000, 8)).astype(np.float32)
    queries = rng.normal(size=(20, 8)).astype(np.float32)
    rows = benchmark_size(vectors, queries, log=lambda _: None)
    exact = next(row for row in rows if row["index"] == "exact")
    assert exact["recall@100"] == 1.0
    assert all(0.0 <= row["recall@100"] <= 1.0 for row in rows)
