# two-tower-recommender: a two-stage recommender, measured against strong baselines

Two-tower retrieval (in-batch softmax, logQ correction, item bias), FAISS, LightGBM re-ranking and a FastAPI service, on MovieLens 1M.

**Author:** Kishor Hange

## Results

Global temporal split (nothing from the future reaches a model), full-catalogue ranking, 1,187 test users.

| Model | NDCG@10 | Recall@100 | Coverage@10 |
|---|---|---|---|
| **EASE (closed-form linear)** | **0.253** | **0.346** | 0.154 |
| Item KNN | 0.243 | 0.297 | 0.074 |
| Two-tower + LightGBM re-ranker | 0.238 | 0.294 | 0.093 |
| Two-tower | 0.226 | 0.294 | 0.077 |
| Popularity | 0.218 | 0.282 | 0.039 |

**The two-tower model lost to EASE.** Paired 95 % intervals: tower minus EASE on NDCG@10 is -0.038 to -0.017, and re-ranked tower minus EASE is -0.025 to -0.005. The re-ranker did help the tower (+0.005 to +0.020). Across 3 seeds the tower's NDCG@10 is 0.221 +/- 0.006. Tuning stopped once the test set had been read, so these numbers weren't chased.

## Ablation

| Change to the tower | NDCG@10 | Recall@100 | Mean popularity percentile of top 10 |
|---|---|---|---|
| Full | 0.226 | 0.294 | 98.8 |
| No logQ correction | 0.036 | 0.073 | 50.6 |
| Next-item (ordered) history | 0.167 | 0.252 | 93.2 |
| With user ID | 0.217 | 0.286 | 99.0 |

- Without logQ, in-batch softmax learns a popularity-free score. The percentile column shows this directly.
- MovieLens users rate in bursts down the site's lists, so rating order partly reflects page layout. Training on a random set of a user's other items beat next-item training.

Only 31 test users were fully cold, too few to draw conclusions.

## The evaluation split changes the winner

The same models on a per-user split, which holds out each user's latest ratings. It's a common choice, but one user's training data can be newer than another's test data.

| Model | NDCG@10 | Recall@100 | Coverage@10 |
|---|---|---|---|
| **Two-tower + LightGBM re-ranker** | **0.105** | 0.411 | 0.345 |
| EASE | 0.099 | **0.450** | 0.236 |
| Two-tower | 0.081 | 0.411 | 0.301 |
| Item KNN | 0.078 | 0.366 | 0.128 |
| Popularity | 0.057 | 0.260 | 0.039 |

Re-ranked tower minus EASE on NDCG@10 is **-0.015 (95 % CI -0.025 to -0.005) on the global split and +0.005 (+0.002 to +0.009) on the per-user split**. Absolute scores can't be compared between the tables, since they cover different users: 1,187 active late in the timeline against 5,820.

Two caveats on reading this as a win for the tower:

- The re-ranker takes EASE's score as a feature, so that row is a hybrid. **The tower alone loses to EASE on both splits.**
- This shows the two splits disagree. It doesn't isolate leakage as the cause, because they also test different users. The global split is the primary result because it matches what a deployed system faces.

## Serving and search speed

Measured one query at a time on one thread (Intel i5-1135G7, 500 queries). A full request (user tower, search, filtering out seen items) on the real catalogue takes **1.4 ms at p50 and 3.0 ms at p99**.

Catalogues above the real 3,678 items are synthetic: noisy copies of the real vectors. They show how search cost scales, not recommendation quality. Recall is against exact search.

| Catalogue | Index | p50 | p99 | Recall@100 | Build |
|---|---|---|---|---|---|
| 3,678 (real) | Exact | 0.10 ms | 0.40 ms | 1.000 | 0 s |
| 3,678 (real) | IVF, 64 probes | 0.10 ms | 0.37 ms | 1.000 | 0.1 s |
| 100,000 | Exact | 3.7 ms | 5.8 ms | 1.000 | 0 s |
| 100,000 | IVF, 64 probes | 0.45 ms | 0.71 ms | 0.997 | 3 s |
| 100,000 | HNSW, ef 256 | 0.28 ms | 0.70 ms | 0.998 | 24 s |
| 1,000,000 | Exact | 34.7 ms | 44.3 ms | 1.000 | 0.1 s |
| 1,000,000 | IVF, 64 probes | 0.88 ms | 1.36 ms | 1.000 | 32 s |
| 1,000,000 | HNSW, ef 512 | 1.65 ms | 2.88 ms | 0.782 | 242 s |

At this catalogue size approximate search gains nothing, so the service uses exact search. At 100,000 items and above IVF at matching recall is about 8 times faster at 100,000 and 40 times at 1M. Don't generalise IVF beating HNSW at 1M: copies with small noise form tight clusters, which suit IVF and are hard for HNSW's graph, and this index searches by inner product rather than a true distance. Real embeddings could rank them differently.

## Run it

```bash
pip install -e ".[dev]"
pytest
towerrec-eval --split global --out results/global
TOWERREC_ARTIFACTS=results/global/artifacts uvicorn towerrec.serving:app --port 8080
```

MovieLens 1M (Harper and Konstan, 2015) is downloaded at run time and not redistributed. Its licence restricts commercial use.
