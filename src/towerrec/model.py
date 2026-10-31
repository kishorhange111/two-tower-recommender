"""Two-tower retrieval model, trained with in-batch softmax and logQ correction.

A positive pair's negatives are the other items in the batch. Popular items therefore appear as negatives
more often than a uniform sample would, and the model learns to push them down. Subtracting log Q(item)
from the logit undoes that sampling bias (Yi et al., RecSys 2019). The ablation measures whether it helps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from .data import AGE_BUCKETS, GENRES, N_OCCUPATIONS, Dataset


@dataclass
class TowerConfig:
    dim: int = 64
    hidden: int = 256
    history_len: int = 50
    use_history: bool = True
    use_features: bool = True
    use_user_id: bool = False  # validation preferred history over a user ID, which only memorises the training period
    log_q: bool = True
    use_item_bias: bool = True  # a cosine can't express "this item is simply popular"; a per-item term can
    history_mode: str = "sampled"  # "sampled": a random set of the user's other items; "before": strictly earlier items
    id_dropout: float = 0.1  # trains the out-of-vocabulary embeddings, which cold users and items rely on
    temperature: float = 0.1
    lr: float = 2e-3
    weight_decay: float = 1e-6
    batch_size: int = 4096
    epochs: int = 15
    patience: int = 3
    seed: int = 0


class TwoTower(nn.Module):
    def __init__(self, data: Dataset, cfg: TowerConfig):
        super().__init__()
        self.cfg = cfg
        self.oov_user = data.n_users
        self.oov_item = data.n_items
        self.user_id = nn.Embedding(data.n_users + 1, cfg.dim)
        self.item_id = nn.Embedding(data.n_items + 2, cfg.dim, padding_idx=data.n_items + 1)
        self.item_bias = nn.Embedding(data.n_items + 1, 1)
        nn.init.zeros_(self.item_bias.weight)
        self.gender = nn.Embedding(2, 8)
        self.age = nn.Embedding(len(AGE_BUCKETS), 8)
        self.occupation = nn.Embedding(N_OCCUPATIONS, 8)
        self.year = nn.Embedding(data.n_year_buckets, 8)
        self.genre = nn.Linear(len(GENRES), 16)
        user_in = cfg.dim + (cfg.dim if cfg.use_history else 0) + (24 if cfg.use_features else 0)
        item_in = cfg.dim + (24 if cfg.use_features else 0)
        self.user_mlp = nn.Sequential(nn.Linear(user_in, cfg.hidden), nn.ReLU(), nn.Linear(cfg.hidden, cfg.dim))
        self.item_mlp = nn.Sequential(nn.Linear(item_in, cfg.hidden), nn.ReLU(), nn.Linear(cfg.hidden, cfg.dim))
        self.register_buffer("gender_values", torch.as_tensor(data.gender))
        self.register_buffer("age_values", torch.as_tensor(data.age))
        self.register_buffer("occupation_values", torch.as_tensor(data.occupation))
        self.register_buffer("genre_values", torch.as_tensor(data.item_genres))
        self.register_buffer("year_values", torch.as_tensor(data.item_year))
        self.register_buffer("user_known", torch.as_tensor(data.user_known))
        self.register_buffer("item_known", torch.as_tensor(data.item_known))

    def _visible(self, ids: torch.Tensor, known: torch.Tensor, oov: int) -> torch.Tensor:
        """Replace IDs with OOV when unseen in training, and at random while training."""
        visible = known[ids]
        if self.training and self.cfg.id_dropout > 0:
            visible = visible & (torch.rand_like(ids, dtype=torch.float32) >= self.cfg.id_dropout)
        return torch.where(visible, ids, torch.full_like(ids, oov))

    def user_vector(self, users: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.cfg.use_user_id:
            parts.append(self.user_id(self._visible(users, self.user_known, self.oov_user)))
        else:
            parts.append(self.user_id(torch.full_like(users, self.oov_user)))
        if self.cfg.use_history:
            mask = (history != self.item_id.padding_idx).unsqueeze(-1)
            # History entries seen only after training fall back to the OOV embedding.
            safe = history.clamp(max=self.oov_item - 1)
            known = self.item_known[safe] | (history == self.item_id.padding_idx)
            embedded = self.item_id(torch.where(known, history, torch.full_like(history, self.oov_item)))
            parts.append((embedded * mask).sum(1) / mask.sum(1).clamp(min=1))
        if self.cfg.use_features:
            parts += [
                self.gender(self.gender_values[users]),
                self.age(self.age_values[users]),
                self.occupation(self.occupation_values[users]),
            ]
        return F.normalize(self.user_mlp(torch.cat(parts, dim=-1)), dim=-1)

    def item_vector(self, items: torch.Tensor) -> torch.Tensor:
        parts = [self.item_id(self._visible(items, self.item_known, self.oov_item))]
        if self.cfg.use_features:
            parts += [self.genre(self.genre_values[items]), self.year(self.year_values[items])]
        return F.normalize(self.item_mlp(torch.cat(parts, dim=-1)), dim=-1)

    def bias(self, items: torch.Tensor) -> torch.Tensor:
        return self.item_bias(self._visible(items, self.item_known, self.oov_item)).squeeze(-1)


class HistoryIndex:
    """Each user's liked items in time order, for building 'history before this moment' without leakage."""

    def __init__(self, positives: pd.DataFrame, n_users: int):
        ordered = positives.sort_values(["user_idx", "ts", "item_idx"], kind="stable")
        self.items: list[np.ndarray] = [np.empty(0, dtype=np.int64) for _ in range(n_users)]
        for user, group in ordered.groupby("user_idx")["item_idx"]:
            self.items[int(user)] = group.to_numpy()

    def latest(self, users: np.ndarray, length: int, pad: int) -> np.ndarray:
        out = np.full((len(users), length), pad, dtype=np.int64)
        for row, user in enumerate(users):
            recent = self.items[int(user)][-length:]
            if len(recent):
                out[row, -len(recent) :] = recent
        return out


def training_examples(positives: pd.DataFrame, history_len: int, pad: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One (user, item, history) row per positive. History holds only strictly earlier positives."""
    ordered = positives.sort_values(["user_idx", "ts", "item_idx"], kind="stable")
    users, items, histories = [], [], []
    for user, group in ordered.groupby("user_idx")["item_idx"]:
        sequence = group.to_numpy()
        matrix = np.full((len(sequence), history_len), pad, dtype=np.int64)
        for position in range(1, len(sequence)):
            recent = sequence[max(0, position - history_len) : position]
            matrix[position, -len(recent) :] = recent
        users.append(np.full(len(sequence), int(user), dtype=np.int64))
        items.append(sequence)
        histories.append(matrix)
    return np.concatenate(users), np.concatenate(items), np.concatenate(histories)


class HistorySampler:
    """Draws, per example, a random set of the user's other training items, excluding the target.

    MovieLens users rate in bursts, working down the lists the site shows them, so the order of ratings within a session
    largely reflects page layout. Predicting "the next item" teaches that layout. Predicting a held-out item from a random
    set of the others teaches taste. All of it stays inside the training fold, so nothing crosses the validation boundary.
    """

    def __init__(self, users: np.ndarray, items: np.ndarray, n_users: int, pad: int):
        order = np.lexsort((items, users))
        self.flat = items[order]
        self.lengths = np.bincount(users, minlength=n_users)
        self.offsets = np.concatenate([[0], np.cumsum(self.lengths)[:-1]])
        self.pad = pad

    def sample(self, users: np.ndarray, targets: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
        lengths = self.lengths[users]
        positions = (rng.random((len(users), length)) * lengths[:, None]).astype(np.int64)
        drawn = self.flat[self.offsets[users][:, None] + positions]
        # A random history size per row, so short and long histories both occur, as they will when serving.
        sizes = (rng.random(len(users)) * np.minimum(lengths - 1, length) + 1).astype(np.int64)
        keep = (np.arange(length)[None, :] < sizes[:, None]) & (drawn != targets[:, None])
        return np.where(keep, drawn, self.pad)


def in_batch_softmax_loss(
    user_vec: torch.Tensor,
    item_vec: torch.Tensor,
    items: torch.Tensor,
    temperature: float,
    log_q: torch.Tensor | None = None,
    item_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Each row's item is its positive; other rows' items are negatives, except repeats of the same item."""
    logits = user_vec @ item_vec.T / temperature
    if item_bias is not None:
        logits = logits + item_bias.unsqueeze(0)
    if log_q is not None:
        logits = logits - log_q[items].unsqueeze(0)
    same_item = items.unsqueeze(0) == items.unsqueeze(1)
    accidental = same_item & ~torch.eye(len(items), dtype=torch.bool, device=items.device)
    logits = logits.masked_fill(accidental, float("-inf"))
    return F.cross_entropy(logits, torch.arange(len(items), device=items.device))


@torch.no_grad()
def embed_items(model: TwoTower, batch: int = 4096) -> np.ndarray:
    """Item vectors for inner-product search. With a bias, it becomes one extra dimension.

    The user side then carries ``[u / temperature, 1]``, so the inner product is exactly the training score
    ``cos(u, v) / temperature + bias``, and a plain maximum-inner-product index can serve it.
    """
    model.eval()
    items = torch.arange(model.oov_item)
    vectors = torch.cat([model.item_vector(chunk) for chunk in items.split(batch)])
    if model.cfg.use_item_bias:
        vectors = torch.cat([vectors, model.bias(items).unsqueeze(-1)], dim=-1)
    return vectors.numpy()


@torch.no_grad()
def embed_users(model: TwoTower, users: np.ndarray, history: HistoryIndex, batch: int = 4096) -> np.ndarray:
    model.eval()
    vectors = []
    pad = model.item_id.padding_idx
    for start in range(0, len(users), batch):
        chunk = users[start : start + batch]
        hist = torch.as_tensor(history.latest(chunk, model.cfg.history_len, pad))
        vector = model.user_vector(torch.as_tensor(chunk), hist)
        if model.cfg.use_item_bias:
            vector = torch.cat([vector / model.cfg.temperature, torch.ones(len(chunk), 1)], dim=-1)
        vectors.append(vector.numpy())
    width = model.cfg.dim + int(model.cfg.use_item_bias)
    return np.concatenate(vectors) if vectors else np.empty((0, width), dtype=np.float32)


def train_two_tower(data: Dataset, cfg: TowerConfig, validate, log=print) -> tuple[TwoTower, dict]:
    """Train with early stopping. ``validate(model) -> float`` scores the current weights, higher is better."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    positives = data.interactions(("train",), positive_only=True)
    model = TwoTower(data, cfg)
    users, items, histories = training_examples(positives, cfg.history_len, model.item_id.padding_idx)
    users_t, items_t, histories_t = map(torch.as_tensor, (users, items, histories))
    sampler = HistorySampler(users, items, data.n_users, model.item_id.padding_idx) if cfg.history_mode == "sampled" else None

    counts = np.bincount(items, minlength=data.n_items).astype(np.float64) + 1.0
    log_q = torch.as_tensor(np.log(counts / counts.sum()), dtype=torch.float32) if cfg.log_q else None
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_score, best_state, best_epoch, curve = -np.inf, None, 0, []
    for epoch in range(cfg.epochs):
        model.train()
        order = rng.permutation(len(users))
        total = 0.0
        batches = 0
        for start in range(0, len(order) - cfg.batch_size + 1, cfg.batch_size):
            index = torch.as_tensor(order[start : start + cfg.batch_size])
            if sampler is None:
                history = histories_t[index]
            else:
                history = torch.as_tensor(sampler.sample(users[index.numpy()], items[index.numpy()], cfg.history_len, rng))
            user_vec = model.user_vector(users_t[index], history)
            item_vec = model.item_vector(items_t[index])
            bias = model.bias(items_t[index]) if cfg.use_item_bias else None
            loss = in_batch_softmax_loss(user_vec, item_vec, items_t[index], cfg.temperature, log_q, bias)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            total += loss.item()
            batches += 1
        score = validate(model)
        curve.append({"epoch": epoch + 1, "loss": total / max(batches, 1), "val": score})
        log(f"  epoch {epoch + 1:2d} loss {total / max(batches, 1):.4f} val {score:.4f}")
        if score > best_score:
            best_score, best_epoch = score, epoch + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif epoch + 1 - best_epoch >= cfg.patience:
            break
    model.load_state_dict(best_state)
    return model, {"config": asdict(cfg), "best_epoch": best_epoch, "best_val": best_score, "curve": curve}
