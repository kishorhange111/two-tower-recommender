"""MovieLens 1M: download, parse, split in time and encode.

Two splits are supported because the choice changes the answer:

- ``global``: one cut-off in time for everyone. Nothing from the future can reach a model, and users
  who first appear after the cut-off are truly cold. This is what a deployed system experiences.
- ``per_user``: each user's last interactions are held out. It is the common choice in papers, but a
  user's training data can be newer than another user's test data, so it leaks across the timeline.

The dataset licence forbids redistribution, so the files are downloaded at run time and never committed.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import scipy.sparse as sp

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
POSITIVE_RATING = 4  # MovieLens convention: 4 and 5 stars count as "liked"

GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]  # fmt: skip
AGE_BUCKETS = [1, 18, 25, 35, 45, 50, 56]
N_OCCUPATIONS = 21


def download(data_dir: str | Path) -> Path:
    """Fetch and unpack the dataset unless it is already there."""
    root = Path(data_dir) / "ml-1m"
    if (root / "ratings.dat").exists():
        return root
    root.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(URL, timeout=120) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(root.parent)
    return root


def load_movielens(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = download(data_dir)
    read = {"sep": "::", "engine": "python", "header": None, "encoding": "latin-1"}
    ratings = pd.read_csv(root / "ratings.dat", names=["user", "item", "rating", "ts"], **read)
    users = pd.read_csv(root / "users.dat", names=["user", "gender", "age", "occupation", "zip"], **read)
    movies = pd.read_csv(root / "movies.dat", names=["item", "title", "genres"], **read)
    return ratings, users, movies


def split_global(ratings: pd.DataFrame, val: float = 0.1, test: float = 0.1) -> pd.DataFrame:
    """Add a ``fold`` column using two cut-offs on the global timeline."""
    ordered = ratings.sort_values(["ts", "user", "item"], kind="stable").reset_index(drop=True)
    n = len(ordered)
    fold = np.full(n, "train", dtype=object)
    fold[int(n * (1 - val - test)) :] = "val"
    fold[int(n * (1 - test)) :] = "test"
    ordered["fold"] = fold
    # A tie on the timestamp must not straddle a cut-off, or the same second is both past and future.
    last_of_fold = ordered.groupby("ts")["fold"].transform("last")
    ordered["fold"] = last_of_fold
    return ordered


def split_per_user(ratings: pd.DataFrame, val: float = 0.1, test: float = 0.1) -> pd.DataFrame:
    """Add a ``fold`` column holding out the last fractions of each user's own timeline."""
    ordered = ratings.sort_values(["user", "ts", "item"], kind="stable").reset_index(drop=True)
    grouped = ordered.groupby("user")["ts"]
    position = grouped.cumcount()
    size = grouped.transform("size")
    fraction = (position + 1) / size
    fold = np.where(fraction > 1 - test, "test", np.where(fraction > 1 - val - test, "val", "train"))
    ordered["fold"] = fold
    return ordered


def make_split(ratings: pd.DataFrame, kind: str) -> pd.DataFrame:
    if kind == "global":
        return split_global(ratings)
    if kind == "per_user":
        return split_per_user(ratings)
    raise ValueError(f"unknown split: {kind}")


@dataclass
class Dataset:
    """Integer-encoded interactions and side features.

    Users and items that first appear after the training fold still get an index, since a real system
    knows they exist, but ``user_known`` / ``item_known`` record that no weights were learned for them.
    The catalogue is every item seen in train or val: an item first seen in test could not be served.
    """

    frame: pd.DataFrame  # user_idx, item_idx, rating, ts, fold
    n_users: int
    n_items: int
    user_known: np.ndarray
    item_known: np.ndarray
    gender: np.ndarray
    age: np.ndarray
    occupation: np.ndarray
    item_genres: np.ndarray  # (n_items, n_genres) multi-hot
    item_year: np.ndarray  # bucket index
    n_year_buckets: int
    item_titles: list[str]
    user_ids: np.ndarray
    item_ids: np.ndarray
    dropped_test_rows: int

    def interactions(self, folds: tuple[str, ...], positive_only: bool = False) -> pd.DataFrame:
        rows = self.frame[self.frame["fold"].isin(folds)]
        return rows[rows["rating"] >= POSITIVE_RATING] if positive_only else rows

    def matrix(self, folds: tuple[str, ...], positive_only: bool = False) -> sp.csr_matrix:
        """Binary user-item matrix for the given folds."""
        rows = self.interactions(folds, positive_only)
        data = np.ones(len(rows), dtype=np.float32)
        csr = sp.csr_matrix((data, (rows["user_idx"], rows["item_idx"])), shape=(self.n_users, self.n_items))
        csr.data[:] = 1.0
        return csr

    def truth(self, fold: str, seen: sp.csr_matrix) -> dict[int, set[int]]:
        """Liked items per user in ``fold``, without anything the user had already interacted with."""
        result: dict[int, set[int]] = {}
        rows = self.interactions((fold,), positive_only=True)
        for user, item in zip(rows["user_idx"].to_numpy(), rows["item_idx"].to_numpy()):
            if not seen[user, item]:
                result.setdefault(int(user), set()).add(int(item))
        return result


def encode(split: pd.DataFrame, users: pd.DataFrame, movies: pd.DataFrame) -> Dataset:
    known_rows = split[split["fold"] != "test"]
    train_rows = split[split["fold"] == "train"]

    item_ids = np.sort(known_rows["item"].unique())
    item_index = {item: i for i, item in enumerate(item_ids)}
    user_ids = np.sort(split["user"].unique())
    user_index = {user: i for i, user in enumerate(user_ids)}

    in_catalogue = split["item"].isin(item_index)
    dropped = int(((split["fold"] == "test") & ~in_catalogue).sum())
    frame = split[in_catalogue].copy()
    frame["user_idx"] = frame["user"].map(user_index).astype(np.int64)
    frame["item_idx"] = frame["item"].map(item_index).astype(np.int64)

    user_table = users.set_index("user").loc[user_ids]
    gender = (user_table["gender"] == "F").to_numpy().astype(np.int64)
    age_lookup = {age: i for i, age in enumerate(AGE_BUCKETS)}
    age = user_table["age"].map(age_lookup).to_numpy().astype(np.int64)
    occupation = user_table["occupation"].to_numpy().astype(np.int64)

    movie_table = movies.set_index("item").loc[item_ids]
    genres = np.zeros((len(item_ids), len(GENRES)), dtype=np.float32)
    genre_index = {genre: i for i, genre in enumerate(GENRES)}
    for row, value in enumerate(movie_table["genres"]):
        for genre in value.split("|"):
            genres[row, genre_index[genre]] = 1.0
    years = movie_table["title"].str.extract(r"\((\d{4})\)\s*$")[0].astype(float).fillna(1995).to_numpy()
    edges = np.array([1950, 1970, 1980, 1985, 1990, 1993, 1995, 1996, 1997, 1998, 1999])
    year_bucket = np.searchsorted(edges, years, side="right").astype(np.int64)

    return Dataset(
        frame=frame[["user_idx", "item_idx", "rating", "ts", "fold"]].reset_index(drop=True),
        n_users=len(user_ids),
        n_items=len(item_ids),
        user_known=np.isin(user_ids, train_rows["user"].unique()),
        item_known=np.isin(item_ids, train_rows["item"].unique()),
        gender=gender,
        age=age,
        occupation=occupation,
        item_genres=genres,
        item_year=year_bucket,
        n_year_buckets=len(edges) + 1,
        item_titles=movie_table["title"].tolist(),
        user_ids=user_ids,
        item_ids=item_ids,
        dropped_test_rows=dropped,
    )
