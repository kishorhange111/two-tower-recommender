"""Online serving: embed the user, search the index, drop what they've seen, and answer.

The item index is computed ahead of time. The user vector is computed per request from the user's latest activity, so a
rating made a moment ago changes the next recommendation with no retraining. That's the operational reason to use a
two-tower model over a matrix factorisation that needs a refit to learn anything about a user.

    TOWERREC_ARTIFACTS=results/global/artifacts uvicorn towerrec.serving:app --port 8080
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from dataclasses import fields
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .data import Dataset
from .model import HistoryIndex, TowerConfig, TwoTower, embed_items


class Recommender:
    def __init__(self, model: TwoTower, data: Dataset, history: HistoryIndex, seen: sp.csr_matrix, index: faiss.Index | None = None):
        self.model = model.eval()
        self.data = data
        self.history = history
        self.seen = seen
        self.user_index = {int(user): i for i, user in enumerate(data.user_ids)}
        self.item_index = {int(item): i for i, item in enumerate(data.item_ids)}
        self.item_vectors = embed_items(model).astype(np.float32)
        if index is None:
            index = faiss.IndexFlatIP(self.item_vectors.shape[1])
            index.add(self.item_vectors)
        self.index = index

    def _user_vector(self, user: int, recent: list[int]) -> np.ndarray:
        cfg = self.model.cfg
        pad = self.model.item_id.padding_idx
        items = np.concatenate([self.history.items[user], np.array(recent, dtype=np.int64)])[-cfg.history_len :]
        padded = np.full((1, cfg.history_len), pad, dtype=np.int64)
        if len(items):
            padded[0, -len(items) :] = items
        with torch.no_grad():
            vector = self.model.user_vector(torch.tensor([user]), torch.as_tensor(padded))
            if cfg.use_item_bias:
                vector = torch.cat([vector / cfg.temperature, torch.ones(1, 1)], dim=-1)
        return vector.numpy().astype(np.float32)

    def recommend(self, user_id: int, k: int = 10, recent_items: list[int] | None = None) -> list[dict]:
        """Top-k unseen items. Raises KeyError for an unknown user. Unknown recent items are ignored."""
        user = self.user_index[int(user_id)]
        recent = [self.item_index[i] for i in (recent_items or []) if i in self.item_index]
        seen = set(self.seen.indices[self.seen.indptr[user] : self.seen.indptr[user + 1]].tolist()) | set(recent)
        # Over-fetch, since filtering happens after the search. Everything seen could rank above everything unseen.
        fetch = min(k + len(seen), self.data.n_items)
        scores, items = self.index.search(self._user_vector(user, recent), fetch)
        results = []
        for score, item in zip(scores[0], items[0]):
            if item < 0 or int(item) in seen:
                continue
            results.append(
                {"item_id": int(self.data.item_ids[item]), "title": self.data.item_titles[item], "score": round(float(score), 4)}
            )
            if len(results) == k:
                break
        return results


def export(model: TwoTower, data: Dataset, folds: tuple[str, ...], directory: str | Path) -> None:
    """Write everything the service needs, so it starts without the raw dataset or the training code path."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), target / "tower.pt")
    (target / "config.json").write_text(json.dumps({f.name: getattr(model.cfg, f.name) for f in fields(TowerConfig)}))
    data.frame[data.frame["fold"].isin(folds)].to_parquet(target / "interactions.parquet")
    np.savez(
        target / "features.npz",
        user_known=data.user_known,
        item_known=data.item_known,
        gender=data.gender,
        age=data.age,
        occupation=data.occupation,
        item_genres=data.item_genres,
        item_year=data.item_year,
        user_ids=data.user_ids,
        item_ids=data.item_ids,
        n_year_buckets=data.n_year_buckets,
    )
    (target / "titles.json").write_text(json.dumps(data.item_titles))


def load(directory: str | Path) -> Recommender:
    target = Path(directory)
    arrays = np.load(target / "features.npz")
    frame = pd.read_parquet(target / "interactions.parquet")
    data = Dataset(
        frame=frame,
        n_users=len(arrays["user_ids"]),
        n_items=len(arrays["item_ids"]),
        user_known=arrays["user_known"],
        item_known=arrays["item_known"],
        gender=arrays["gender"],
        age=arrays["age"],
        occupation=arrays["occupation"],
        item_genres=arrays["item_genres"],
        item_year=arrays["item_year"],
        n_year_buckets=int(arrays["n_year_buckets"]),
        item_titles=json.loads((target / "titles.json").read_text()),
        user_ids=arrays["user_ids"],
        item_ids=arrays["item_ids"],
        dropped_test_rows=0,
    )
    cfg = TowerConfig(**json.loads((target / "config.json").read_text()))
    model = TwoTower(data, cfg)
    model.load_state_dict(torch.load(target / "tower.pt", weights_only=True))
    folds = tuple(frame["fold"].unique())
    return Recommender(model, data, HistoryIndex(data.interactions(folds, positive_only=True), data.n_users), data.matrix(folds))


class RecommendRequest(BaseModel):
    user_id: int
    k: int = Field(default=10, ge=1, le=100)
    recent_items: list[int] = Field(default_factory=list, max_length=200)


def create_app(recommender: Recommender | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.recommender = recommender if recommender is not None else load(os.environ.get("TOWERREC_ARTIFACTS", "artifacts"))
        yield

    app = FastAPI(title="Recommender", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request) -> dict:
        service = request.app.state.recommender
        return {"status": "ok", "users": service.data.n_users, "items": service.data.n_items}

    @app.post("/recommend")
    def recommend(body: RecommendRequest, request: Request) -> dict:
        started = time.perf_counter()
        try:
            items = request.app.state.recommender.recommend(body.user_id, body.k, body.recent_items)
        except KeyError as error:
            raise HTTPException(404, f"unknown user: {body.user_id}") from error
        return {"user_id": body.user_id, "items": items, "latency_ms": round((time.perf_counter() - started) * 1000, 2)}

    return app


# Built at import for uvicorn. Loading waits for start-up, so importing this module stays cheap.
app = create_app()
