"""
High-level experiment runners: same procedure for every model, results as tables.
Notebook cells call these and display the output, no training logic in notebooks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation import compute_threshold, first_confirmed_alarm, false_positive_rate


def _alarm_str(alarm) -> str | None:
    return alarm.strftime("%Y-%m-%d %H:%M:%S") if alarm is not None else None


def run_classical_suite(models, X_healthy_train, X_all, X_healthy_val,
                        index, train_end, val_end, verbose=False) -> tuple[pd.DataFrame, dict]:
    """
    Fit each classical detector on healthy train only, threshold on validation
    scores, alarm and FPR on the full timeline. Returns summary table plus a
    dict of full scores and thresholds per model for downstream plots.
    """
    results = {}
    for name, fit_fn, score_fn in models:
        if verbose:
            print(f"Training {name}...", flush=True)
        model = fit_fn(X_healthy_train)
        full_scores = score_fn(model, X_all)
        val_scores = score_fn(model, X_healthy_val)
        val_mean = float(np.mean(val_scores))
        val_std = float(np.std(val_scores))
        scores_series = pd.Series(full_scores, index=index)
        threshold = compute_threshold(val_scores, method="mean_std", n_std=3.0)
        is_anomaly = scores_series > threshold
        alarm = first_confirmed_alarm(is_anomaly, window=20)
        fpr_train = float(is_anomaly[index <= train_end].mean())
        fpr_val = float(is_anomaly[(index > train_end) & (index <= val_end)].mean())
        results[name] = {"scores": scores_series, "val_mean": val_mean, "val_std": val_std,
                         "threshold": float(threshold), "alarm": alarm,
                         "fpr_train": fpr_train, "fpr_val": fpr_val}

    summary = pd.DataFrame([{
        "model": name,
        "val_mean": r["val_mean"],
        "val_std": r["val_std"],
        "threshold": r["threshold"],
        "first_alarm": r["alarm"],
        "fpr_train": r["fpr_train"],
        "fpr_val": r["fpr_val"],
    } for name, r in results.items()]).sort_values("model").reset_index(drop=True)
    return summary, results


def _score_setup(models, Xtr, Xa_, Xva, index, train_end, val_end,
                   label: str | None = None) -> list[dict]:
    rows = []
    for name, fit_fn, score_fn in models:
        if label is not None:
            print(f"Training {name} ({label})...", flush=True)
        model = fit_fn(Xtr)
        full = pd.Series(score_fn(model, Xa_), index=index)
        val = score_fn(model, Xva)
        threshold = compute_threshold(val, method="mean_std", n_std=3.0)
        is_an = full > threshold
        alarm = first_confirmed_alarm(is_an, window=20)
        fpr_train = float(is_an[(index <= train_end)].mean())
        fpr_val = float(is_an[(index > train_end) & (index <= val_end)].mean())
        rows.append({"model": name, "threshold": float(threshold),
                     "alarm": _alarm_str(alarm),
                     "fpr_train": round(fpr_train, 4), "fpr_val": round(fpr_val, 4)})
    return rows


def compare_column_setups(models, setups: dict[str, list[str]], df: pd.DataFrame,
                          df_healthy_train: pd.DataFrame, df_healthy_val: pd.DataFrame,
                          train_end, val_end, verbose=False) -> pd.DataFrame:
    """
    Same procedure on different column subsets (feature types or bearings).
    setups maps setup name to column list. Scaler is fit per subset on healthy train.
    Returns long table with setup, model, threshold, alarm, fpr_train, fpr_val.
    """
    from sklearn.preprocessing import MinMaxScaler

    rows = []
    for setup, cols in setups.items():
        if verbose:
            print(f"--- {setup} ---", flush=True)
        scaler = MinMaxScaler()
        Xtr = scaler.fit_transform(df_healthy_train[cols])
        Xa_ = scaler.transform(df[cols])
        Xva = scaler.transform(df_healthy_val[cols])
        for row in _score_setup(models, Xtr, Xa_, Xva, df.index, train_end, val_end):
            rows.append({"setup": setup, **row})
    return pd.DataFrame(rows)


def boundary_sensitivity(models, boundaries: dict[str, pd.Timestamp], feature_cols: list[str],
                         df: pd.DataFrame, val_fraction: float = 0.2,
                         verbose=False) -> pd.DataFrame:
    """
    Repeat the full procedure (split, scale, fit, threshold) per boundary.
    Uses split_train_val with val_fraction on each healthy period.
    """
    from src.preprocessing import split_healthy, split_train_val, scale_features

    rows = []
    for label, b in boundaries.items():
        if verbose:
            print(f"Boundary {label}: training 4 models...", flush=True)
        df_h = split_healthy(df[feature_cols], b)
        df_tr, df_va = split_train_val(df_h, val_fraction=val_fraction)
        Xtr, Xa_, scaler = scale_features(df_tr, df[feature_cols])
        Xva = scaler.transform(df_va)
        train_end, val_end = df_tr.index[-1], df_h.index[-1]
        for row in _score_setup(models, Xtr, Xa_, Xva, df.index, train_end, val_end):
            rows.append({"boundary_label": label, **row})
    return pd.DataFrame(rows)


def compare_scalers(models, scalers: dict[str, type], df: pd.DataFrame,
                    df_healthy_train: pd.DataFrame, df_healthy_val: pd.DataFrame,
                    train_end, val_end, verbose=False) -> pd.DataFrame:
    """
    Same procedure under different scalers (classes, instantiated per use).
    Returns long table with scaler, model, threshold, alarm, fpr_train, fpr_val.
    """
    rows = []
    for sname, scaler_cls in scalers.items():
        scaler = scaler_cls()
        Xtr = scaler.fit_transform(df_healthy_train)
        Xa_ = scaler.transform(df)
        Xva = scaler.transform(df_healthy_val)
        for row in _score_setup(models, Xtr, Xa_, Xva, df.index, train_end, val_end,
                                label=sname):
            rows.append({"scaler": sname, **row})
    return pd.DataFrame(rows)
