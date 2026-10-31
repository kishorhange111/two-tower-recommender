"""When does approximate search earn its keep?

MovieLens has about 3,700 items, far below the size where ANN matters, so a benchmark on the real catalogue alone would say
nothing useful. To find the crossover, the real item vectors are copied with Gaussian noise up to 1M items. Those larger
catalogues are synthetic: they show how search cost scales, not recommendation quality. Recall is measured against exact search
on the same catalogue, and latency is one query at a time on one thread, as a single request sees it.

    towerrec-ann --artifacts results/global/artifacts --out results/ann.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np
import torch

from .serving import load

K = 100


def scale_catalogue(vectors: np.ndarray, size: int, noise: float = 0.02, seed: int = 0) -> np.ndarray:
    """Repeat the vectors with noise until there are ``size`` of them. The originals stay at the front."""
    if size <= len(vectors):
        return vectors[:size]
    rng = np.random.default_rng(seed)
    repeats = int(np.ceil(size / len(vectors)))
    copies = [vectors] + [vectors + rng.normal(0, noise, vectors.shape).astype(np.float32) for _ in range(repeats - 1)]
    return np.concatenate(copies)[:size]


def timed_search(index: faiss.Index, queries: np.ndarray, exact: np.ndarray) -> dict:
    latencies = []
    found = []
    for query in queries:
        start = time.perf_counter()
        _, items = index.search(query[None, :], K)
        latencies.append((time.perf_counter() - start) * 1000)
        found.append(items[0])
    recall = np.mean([len(set(a.tolist()) & set(b.tolist())) / K for a, b in zip(found, exact)])
    return {"p50_ms": float(np.percentile(latencies, 50)), "p99_ms": float(np.percentile(latencies, 99)), "recall@100": float(recall)}


def benchmark_size(vectors: np.ndarray, queries: np.ndarray, log) -> list[dict]:
    dim = vectors.shape[1]
    n = len(vectors)
    rows = []

    def record(name: str, build_seconds: float, index: faiss.Index, exact: np.ndarray) -> None:
        row = {"catalogue": n, "index": name, "build_seconds": round(build_seconds, 2), **timed_search(index, queries, exact)}
        rows.append(row)
        log(
            f"  {name:22s} build {build_seconds:7.1f}s p50 {row['p50_ms']:8.3f}ms p99 {row['p99_ms']:8.3f}ms recall@100 {row['recall@100']:.4f}"
        )

    start = time.perf_counter()
    flat = faiss.IndexFlatIP(dim)
    flat.add(vectors)
    build = time.perf_counter() - start
    _, exact = flat.search(queries, K)
    record("exact", build, flat, exact)

    start = time.perf_counter()
    hnsw = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 100
    hnsw.add(vectors)
    build = time.perf_counter() - start
    for ef in (128, 256, 512):
        hnsw.hnsw.efSearch = ef
        record(f"hnsw_m32_ef{ef}", build, hnsw, exact)

    nlist = max(8, int(4 * np.sqrt(n)))
    start = time.perf_counter()
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, nlist, faiss.METRIC_INNER_PRODUCT)
    rng = np.random.default_rng(0)
    train_size = min(n, max(nlist * 40, 10_000))
    ivf.train(vectors[rng.choice(n, train_size, replace=False)])
    ivf.add(vectors)
    build = time.perf_counter() - start
    for nprobe in (4, 16, 64):
        ivf.nprobe = nprobe
        record(f"ivf_{nlist}_probe{nprobe}", build, ivf, exact)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts", default="results/global/artifacts")
    parser.add_argument("--out", default="results/ann.json")
    parser.add_argument("--sizes", nargs="*", type=int, default=[0, 100_000, 1_000_000], help="0 means the real catalogue")
    parser.add_argument("--queries", type=int, default=500)
    args = parser.parse_args()

    def log(message: str) -> None:
        print(message, flush=True)

    faiss.omp_set_num_threads(1)
    recommender = load(args.artifacts)
    rng = np.random.default_rng(0)
    users = rng.choice(recommender.data.n_users, size=args.queries, replace=False)
    queries = np.concatenate([recommender._user_vector(int(u), []) for u in users])
    torch.set_num_threads(1)
    latencies = []
    for user in recommender.data.user_ids[users]:
        start = time.perf_counter()
        recommender.recommend(int(user), k=10)
        latencies.append((time.perf_counter() - start) * 1000)
    # The whole request path on the real catalogue: user tower, exact search and filtering out seen items.
    end_to_end = {"p50_ms": float(np.percentile(latencies, 50)), "p99_ms": float(np.percentile(latencies, 99))}
    log(f"end to end on the real catalogue: {end_to_end}")
    results = []
    for size in args.sizes:
        vectors = recommender.item_vectors if size == 0 else scale_catalogue(recommender.item_vectors, size)
        log(f"catalogue of {len(vectors):,} items")
        results += benchmark_size(vectors, queries, log)
    report = {
        "k": K,
        "threads": 1,
        "queries": args.queries,
        "real_catalogue": recommender.data.n_items,
        "synthetic_above_real": True,
        "end_to_end_real_catalogue": end_to_end,
        "rows": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
