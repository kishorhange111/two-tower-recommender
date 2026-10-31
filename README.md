# two-tower-recommender: a two-stage recommender, measured against strong baselines

Two-tower retrieval (in-batch softmax, logQ correction, item bias), FAISS, LightGBM re-ranking and a FastAPI service, on MovieLens 1M.

**Author:** Kishor Hange

## Results

Global temporal split (nothing from the future reaches a model), full-catalogue ranking, 1,187 test users.

| Model | NDCG@10 | Recall@100 | Coverage@10 |
|---|---|---|---|
| **EASE (closed-form linear)** | **0.253** | **0.346** | 0.154 |
| Two-tower + LightGBM re-ranker | 0.238 | 0.294 | 0.093 |
| Item KNN | 0.243 | 0.297 | 0.074 |
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

Only 31 test users were fully cold, too few to draw conclusions. The approximate-search benchmark (`towerrec-ann`) and the per-user split comparison are implemented, but their results aren't reported here.

## Run it

```bash
pip install -e ".[dev]"
pytest
towerrec-eval --split global --out results/global
TOWERREC_ARTIFACTS=results/global/artifacts uvicorn towerrec.serving:app --port 8080
```

MovieLens 1M (Harper and Konstan, 2015) is downloaded at run time and not redistributed. Its licence restricts commercial use.
