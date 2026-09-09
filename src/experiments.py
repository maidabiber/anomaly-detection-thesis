"""
High-level experiment runners: same procedure for every model, results as tables.
Notebook cells call these and display the output, no training logic in notebooks.
"""
import numpy as np
import pandas as pd

from src.evaluation import compute_threshold, first_confirmed_alarm, false_positive_rate


def run_classical_suite(models, X_healthy_train, X_all, X_healthy_val,
                        index, train_end, val_end) -> tuple[pd.DataFrame, dict]:
    """
    Fit each classical detector on healthy train only, threshold on validation
    scores, alarm and FPR on the full timeline. Returns summary table plus a
    dict of full scores and thresholds per model for downstream plots.
    """
    results = {}
    for name, fit_fn, score_fn in models:
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
