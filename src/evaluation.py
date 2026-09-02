from __future__ import annotations

import numpy as np
import pandas as pd


def compute_threshold(healthy_scores: np.ndarray, method: str = "mean_std", n_std: float = 3.0,
                       percentile: float = 99.5) -> float:
    if method == "mean_std":
        return float(np.mean(healthy_scores) + n_std * np.std(healthy_scores))
    elif method == "percentile":
        return float(np.percentile(healthy_scores, percentile))
    else:
        raise ValueError(f"Unknown method: {method}")


def apply_continuity_filter(is_anomaly: pd.Series, window: int = 20) -> pd.Series:
    return is_anomaly.rolling(window=window).sum() == window


def first_confirmed_alarm(is_anomaly: pd.Series, window: int = 20) -> pd.Timestamp | None:
    confirmed = apply_continuity_filter(is_anomaly, window=window)
    confirmed_true = confirmed[confirmed]
    if len(confirmed_true) == 0:
        return None
    return confirmed_true.index[0]


def false_positive_rate(is_anomaly: pd.Series, healthy_period_end: pd.Timestamp) -> float:
    healthy_flags = is_anomaly[is_anomaly.index <= healthy_period_end]
    if len(healthy_flags) == 0:
        return float("nan")
    return float(healthy_flags.mean())


def detection_delay(first_alarm: pd.Timestamp, known_fault_start: pd.Timestamp) -> pd.Timedelta:
    return first_alarm - known_fault_start


def precision_recall_f1(is_anomaly: pd.Series, known_fault_start: pd.Timestamp,
                         window: int = 20) -> dict:
    predicted = apply_continuity_filter(is_anomaly, window=window).fillna(False)
    actual = pd.Series(is_anomaly.index >= known_fault_start, index=is_anomaly.index)

    tp = int((predicted & actual).sum())
    fp = int((predicted & ~actual).sum())
    fn = int((~predicted & actual).sum())
    tn = int((~predicted & ~actual).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "true_positives": tp, "false_positives": fp,
        "false_negatives": fn, "true_negatives": tn,
    }