"""
Disk cache for trained scores / result tables.

Motivation: deep suites (run_deep_suite, sweep_deep_training_params,
deep_boundary_sensitivity, CNN) cost minutes-hours. Re-running the whole
notebook just to render the final comparison table is wasteful.

What is cached: plain python objects (dicts of scores, DataFrames as
parquet/csv fallback to pickle). What is NOT cached: keras Model objects
(weights) - we store the aggregated full/val score arrays + thresholds,
which is everything the final table / sweeps need. Retraining from the
same cache key reproduces identical numbers (TF_DETERMINISTIC_OPS +
fixed SEED_START in src/deep_models.py).

Cache key = explicit string (e.g. "3rd_test_v2/winning_classical") plus a
sidecar .meta.json with:
  - data_hash (sha1 over input arrays, same helper as experiments.py)
  - config snapshot (EPOCHS, N_RUNS, BATCH_SIZE, THRESHOLD_*, WINDOW_SIZE,
    CONTINUITY_WINDOW)
  - boundary / extra label string supplied by the caller

On load the meta is compared; on mismatch the entry is ignored (never
silently reuse stale scores). Classical models train in seconds, so
caching them is only for making the final cell standalone - the real
win is deep + CNN.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle

import numpy as np

CACHE_ROOT = os.path.join("results", "cache")


def _config_snapshot() -> dict:
    import src.config as config
    keys = ["EPOCHS", "BATCH_SIZE", "VAL_FRACTION", "THRESHOLD_METHOD",
            "THRESHOLD_N_STD", "THRESHOLD_PERCENTILE", "CONTINUITY_WINDOW",
            "WINDOW_SIZE", "ENCODING_DIM", "N_RUNS", "SEED_START"]
    return {k: getattr(config, k, None) for k in keys}


def hash_arrays(*arrays) -> str:
    h = hashlib.sha1()
    for a in arrays:
        arr = np.asarray(a)
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def _paths(name: str) -> tuple[str, str]:
    base = os.path.join(CACHE_ROOT, name)
    return base + ".pkl", base + ".meta.json"


def save(name: str, obj, data_hash: str | None = None, extra: dict | None = None) -> str:
    """Pickle obj to results/cache/<name>.pkl + write .meta.json. Returns path."""
    pkl_path, meta_path = _paths(name)
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    meta = {"config": _config_snapshot(), "data_hash": data_hash, "extra": extra or {}}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, default=str)
    return pkl_path


def load(name: str, data_hash: str | None = None, extra: dict | None = None):
    """
    Load cached object or return None.
    Returns None when: file missing, unpickling fails, config snapshot
    differs, data_hash differs (when given), or extra dict differs.
    Never raises on stale cache - caller falls back to training.
    """
    pkl_path, meta_path = _paths(name)
    if not (os.path.exists(pkl_path) and os.path.exists(meta_path)):
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return None
    if meta.get("config") != _config_snapshot():
        return None
    if data_hash is not None and meta.get("data_hash") != data_hash:
        return None
    if extra is not None and meta.get("extra") != extra:
        return None
    try:
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None
