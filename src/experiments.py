"""
High-level experiment runners: same procedure for every model, results as tables.
Notebook cells call these and display the output, no training logic in notebooks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import src.config as config
from src.evaluation import (compute_threshold, first_confirmed_alarm, false_positive_rate,
                            classify_alarm, score_detection)


def _alarm_str(alarm) -> str | None:
    return alarm.strftime("%Y-%m-%d %H:%M:%S") if alarm is not None else None


def _hash_arrays(*arrays) -> str:
    """Short sha1 over array bytes+shapes. Guards cache reuse against changed inputs."""
    import hashlib
    h = hashlib.sha1()
    for a in arrays:
        arr = np.asarray(a)
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def _threshold_alarm_fpr(scores_series: pd.Series, val_scores, index, train_end, val_end, window=None):
    """Shared threshold (mean+3std), alarm (window) and FPRs. One place, used everywhere."""
    val_mean = float(np.mean(val_scores))
    val_std = float(np.std(val_scores))
    threshold = compute_threshold(val_scores, method=config.THRESHOLD_METHOD, n_std=config.THRESHOLD_N_STD,
                                      percentile=config.THRESHOLD_PERCENTILE)
    is_anomaly = scores_series > threshold
    alarm = first_confirmed_alarm(is_anomaly,
                                  window=(window if window is not None else config.CONTINUITY_WINDOW))
    fpr_train = float(is_anomaly[index <= train_end].mean())
    fpr_val = float(is_anomaly[(index > train_end) & (index <= val_end)].mean())
    return threshold, alarm, fpr_train, fpr_val, val_mean, val_std


def run_classical_suite(models, X_healthy_train, X_all, X_healthy_val,
                        index, train_end, val_end, verbose=False,
                        window=None) -> tuple[pd.DataFrame, dict]:
    """
    Fit each classical detector on healthy train only, threshold on validation
    scores, alarm and FPR on the full timeline. Returns summary table plus a
    dict of full scores and thresholds per model for downstream plots.
    window: continuity window for the confirmed alarm (default None means
    config.CONTINUITY_WINDOW). Threshold always follows config.py defaults.
    """
    results = {}
    for name, fit_fn, score_fn in models:
        if verbose:
            print(f"Training {name}...", flush=True)
        model = fit_fn(X_healthy_train)
        full_scores = score_fn(model, X_all)
        val_scores = score_fn(model, X_healthy_val)
        scores_series = pd.Series(full_scores, index=index)
        threshold, alarm, fpr_train, fpr_val, val_mean, val_std = _threshold_alarm_fpr(
            scores_series, val_scores, index, train_end, val_end, window=window)
        results[name] = {"scores": scores_series, "val_scores": np.asarray(val_scores),
                         "val_mean": val_mean, "val_std": val_std,
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
                   label: str | None = None, window=None) -> list[dict]:
    rows = []
    for name, fit_fn, score_fn in models:
        if label is not None:
            print(f"Training {name} ({label})...", flush=True)
        model = fit_fn(Xtr)
        full = pd.Series(score_fn(model, Xa_), index=index)
        val = score_fn(model, Xva)
        threshold, alarm, fpr_train, fpr_val, _, _ = _threshold_alarm_fpr(
            full, val, index, train_end, val_end, window=window)
        rows.append({"model": name, "threshold": float(threshold),
                     "alarm": _alarm_str(alarm),
                     "fpr_train": round(fpr_train, 4), "fpr_val": round(fpr_val, 4)})
    return rows


def compare_column_setups(models, setups: dict[str, list[str]], df: pd.DataFrame,
                          df_healthy_train: pd.DataFrame, df_healthy_val: pd.DataFrame,
                          train_end, val_end, verbose=False, window=None) -> pd.DataFrame:
    """
    Same procedure on different column subsets (feature types or bearings).
    setups maps setup name to column list. Scaler is fit per subset on healthy train.
    window: continuity window for the confirmed alarm (default None means
    config.CONTINUITY_WINDOW). Threshold always follows config.py defaults.
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
        for row in _score_setup(models, Xtr, Xa_, Xva, df.index, train_end, val_end, window=window):
            rows.append({"setup": setup, **row})
    return pd.DataFrame(rows)


def compare_feature_setups(models, df: pd.DataFrame, df_healthy_train: pd.DataFrame,
                           df_healthy_val: pd.DataFrame, train_end, val_end,
                           feature_types: list[str] | None = None, verbose=False,
                           window=None) -> pd.DataFrame:
    """
    Joint (all_24 vs all_36) plus per-feature setups in one table.
    Uses OLD6 (24 time features) and all 36 (time+spectral) as two joints,
    then one entry per feature type. Wrapper around compare_column_setups.
    window is passed through (default None means config.CONTINUITY_WINDOW).
    """
    if feature_types is None:
        feature_types = ["RMS", "Kurtosis", "CrestFactor", "Peak", "Skewness", "Std",
                         "SpecCentroid", "HighFreqRatio", "SpecPeak"]
    OLD6 = ["RMS", "Kurtosis", "CrestFactor", "Peak", "Skewness", "Std"]
    old_cols = [c for c in df.columns if any(c.endswith("_" + f) for f in OLD6)]
    setups = {
        "all_24": old_cols,
        "all_36": list(df.columns),
    }
    for feat in feature_types:
        setups[feat] = [c for c in df.columns if c.endswith("_" + feat)]
    return compare_column_setups(models, setups, df, df_healthy_train, df_healthy_val,
                                 train_end, val_end, verbose=verbose, window=window)


def sweep_detector_params(models, X_healthy_train, X_all, X_healthy_val,
                          index, train_end, val_end,
                          windows: tuple = (10, 20, 30),
                          known_fault_start=None, burn_in_end=None,
                          verbose=False, return_fitted=False):
    """
    Fit each model once, then score every window x threshold-rule combo
    from the same scores. One row per (model, window, rule): threshold,
    alarm, and either fpr_val (no onset) or verdict/delay/precision/recall/f1.
    Same table answers "which window, which formula" side by side.
    If return_fitted is True, return (table, fitted) where fitted maps
    model name to the fitted model plus full/val scores and a data hash,
    so compare_joint_24vs36_full_grid(..., cached_36=fitted) can skip
    refitting the identical all_36 setup. Default False keeps the old
    single-DataFrame return value.
    """
    thr_rules = {
        "mean+2std": lambda v: compute_threshold(v, method="mean_std", n_std=2.0),
        "mean+3std": lambda v: compute_threshold(v, method="mean_std", n_std=config.THRESHOLD_N_STD),
        "percentile_99.5": lambda v: compute_threshold(v, method="percentile", percentile=config.THRESHOLD_PERCENTILE),

    }
    rows = []
    fitted = {}
    data_hash = _hash_arrays(X_healthy_train, X_all, X_healthy_val)
    for name, fit_fn, score_fn in models:
        if verbose:
            print(f"Training {name}...", flush=True)
        model = fit_fn(X_healthy_train)
        full = pd.Series(score_fn(model, X_all), index=index)
        val = score_fn(model, X_healthy_val)
        fitted[name] = {"model": model, "full": full, "val": np.asarray(val),
                        "fit_fn": fit_fn, "score_fn": score_fn,
                        "data_hash": data_hash}
        for w in windows:
            for rname, rfn in thr_rules.items():
                t = float(rfn(val))
                is_an = full > t
                alarm = first_confirmed_alarm(is_an, window=w)
                row = {"model": name, "window": w, "rule": rname,
                       "threshold": round(t, 5), "alarm": _alarm_str(alarm)}
                if known_fault_start is not None:
                    v = classify_alarm(alarm, known_fault_start, burn_in_end=burn_in_end)
                    m = score_detection(is_an, known_fault_start,
                                        burn_in_end=burn_in_end, window=w)
                    row.update({"verdict": v["label"],
                                "delay": (str(v["delay"]) if v["delay"] is not None else None),
                                "precision": round(m["precision"], 3),
                                "recall": round(m["recall"], 3),
                                "f1": round(m["f1"], 3)})
                else:
                    fpr = float(is_an[(full.index > train_end) & (full.index <= val_end)].mean())
                    row["fpr_val"] = round(fpr, 4)
                rows.append(row)
    out = pd.DataFrame(rows)
    if return_fitted:
        return out, fitted
    return out


def sweep_training_params(variants, X_healthy_train, X_all, X_healthy_val,
                          index, train_end, val_end,
                          window=None, known_fault_start=None, burn_in_end=None,
                          verbose=False) -> pd.DataFrame:
    """
    Train one detector per hyperparameter variant, score all identically.

    variants: list of (label, fit_fn, score_fn), e.g. from
    classical_models.classical_training_grid(). Threshold recipe is fixed
    (config.THRESHOLD_METHOD); only training differs. One row per variant:
    threshold, alarm, plus fpr_val (no onset) or verdict/delay/P/R/F1.
    """
    w = window if window is not None else config.CONTINUITY_WINDOW
    rows = []
    for label, fit_fn, score_fn in variants:
        if verbose:
            print(f"Training {label}...", flush=True)
        model = fit_fn(X_healthy_train)
        full = pd.Series(score_fn(model, X_all), index=index)
        val = score_fn(model, X_healthy_val)
        threshold = compute_threshold(val, method=config.THRESHOLD_METHOD,
                                      n_std=config.THRESHOLD_N_STD,
                                      percentile=config.THRESHOLD_PERCENTILE)
        is_an = full > threshold
        alarm = first_confirmed_alarm(is_an, window=w)
        row = {"variant": label, "threshold": round(float(threshold), 5),
               "alarm": _alarm_str(alarm)}
        if known_fault_start is not None:
            v = classify_alarm(alarm, known_fault_start, burn_in_end=burn_in_end)
            m = score_detection(is_an, known_fault_start,
                                burn_in_end=burn_in_end, window=w)
            row.update({"verdict": v["label"],
                        "delay": (str(v["delay"]) if v["delay"] is not None else None),
                        "precision": round(m["precision"], 3),
                        "recall": round(m["recall"], 3),
                        "f1": round(m["f1"], 3)})
        else:
            fpr = float(is_an[(full.index > train_end) & (full.index <= val_end)].mean())
            row["fpr_val"] = round(fpr, 4)
        rows.append(row)
    return pd.DataFrame(rows)


def sweep_deep_training_params(variants, X_healthy_train, X_all, X_healthy_val,
                               index, train_end, val_end, input_dim,
                               window_size=10, windows: tuple = (10, 20, 30),
                               known_fault_start=None, burn_in_end=None,
                               verbose=False, return_fitted=False):
    """
    Train one deep detector per (architecture, epochs, aggregation) variant,
    then score every window x threshold-rule combo from the same errors
    (no retraining).
    variants: list of (label, kind, layers, epochs, agg) from
    deep_models.deep_training_grid(); kind is "dense" or "lstm".
    Costs len(variants) x N_RUNS fits.
    One row per (variant, window, rule): threshold, alarm, plus fpr or
    verdict/P/R/F1.
    If return_fitted is True, return (table, fitted) where fitted maps
    variant label to aggregated full/val scores plus ref_index, input_dim,
    window_size, variant spec and a data hash, so
    compare_joint_24vs36_full_grid_deep(..., cached_36=fitted) can skip
    refitting the identical all_36 setup. Default False keeps the old
    single-DataFrame return value.
    """
    import os
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    import tensorflow as tf
    try:
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
    except RuntimeError:
        pass

    from src.deep_models import (
        build_autoencoder, build_lstm_autoencoder, make_windows,
        run_multiple, reconstruction_error, reconstruction_error_lstm,
    )

    thr_rules = {
        "mean+2std": lambda v: compute_threshold(v, method="mean_std", n_std=2.0),
        "mean+3std": lambda v: compute_threshold(v, method="mean_std", n_std=config.THRESHOLD_N_STD),
        "percentile_99.5": lambda v: compute_threshold(v, method="percentile", percentile=config.THRESHOLD_PERCENTILE),

    }
    Xw_tr, Xw_all, Xw_val = None, None, None
    if any(k == "lstm" for _, k, _, _, _ in variants):
        Xw_tr = make_windows(X_healthy_train, window_size)
        Xw_all = make_windows(X_all, window_size)
        Xw_val = make_windows(X_healthy_val, window_size)
        lstm_index = index[window_size - 1:]

    rows = []
    fitted = {}
    data_hash = _hash_arrays(X_healthy_train, X_all, X_healthy_val)
    for label, kind, layers, epochs, agg in variants:
        if verbose:
            print(f"Training {label} ({config.N_RUNS} runs)...", flush=True)
        if kind == "lstm":
            build_fn = lambda: build_lstm_autoencoder(window_size, input_dim,
                                                      encoding_dim=config.ENCODING_DIM)
            (mean_full, _), (mean_val, _) = run_multiple(
                build_fn, Xw_tr, Xw_all, error_fn=reconstruction_error_lstm,
                X_val=Xw_val, method=agg,
                n_runs=config.N_RUNS, epochs=epochs, batch_size=config.BATCH_SIZE)
            scores_series = pd.Series(mean_full, index=lstm_index)
            ref_index = lstm_index
        else:
            build_fn = lambda hl=layers: build_autoencoder(input_dim, hl)
            (mean_full, _), (mean_val, _) = run_multiple(
                build_fn, X_healthy_train, X_all, error_fn=reconstruction_error,
                X_val=X_healthy_val, method=agg, n_runs=config.N_RUNS,
                epochs=epochs, batch_size=config.BATCH_SIZE)
            scores_series = pd.Series(mean_full, index=index)
            ref_index = index
        fitted[label] = {"kind": kind, "layers": layers, "epochs": epochs, "agg": agg,
                         "full": scores_series, "val": np.asarray(mean_val),
                         "ref_index": ref_index, "input_dim": input_dim,
                         "window_size": window_size, "data_hash": data_hash}
        for w in windows:
            for rname, rfn in thr_rules.items():
                t = float(rfn(mean_val))
                is_an = scores_series > t
                alarm = first_confirmed_alarm(is_an, window=w)
                row = {"variant": label, "window": w, "rule": rname,
                       "threshold": round(t, 5), "alarm": _alarm_str(alarm)}
                if known_fault_start is not None:
                    v = classify_alarm(alarm, known_fault_start, burn_in_end=burn_in_end)
                    m = score_detection(is_an, known_fault_start,
                                        burn_in_end=burn_in_end, window=w)
                    row.update({"verdict": v["label"],
                                "delay": (str(v["delay"]) if v["delay"] is not None else None),
                                "precision": round(m["precision"], 3),
                                "recall": round(m["recall"], 3),
                                "f1": round(m["f1"], 3)})
                else:
                    row["fpr_val"] = round(float(is_an[(ref_index > train_end)
                                                       & (ref_index <= val_end)].mean()), 4)
                rows.append(row)
    out = pd.DataFrame(rows)
    if return_fitted:
        return out, fitted
    return out


def sweep_trained_scores(results: dict, train_end, val_end,
                         windows: tuple = (10, 15, 20),
                         known_fault_start=None, burn_in_end=None) -> pd.DataFrame:
    """
    Window x threshold-rule grid over ALREADY trained models (no retraining).

    results: dict from run_classical_suite or run_deep_suite (needs "scores"
    and "val_scores" per model). Deep results also carry scores_mean /
    scores_median from the same runs: both aggregations are swept, one extra
    "agg" column. Same pivot layout as the 8b detector sweep: one row per
    (model, agg, window, rule), alarm plus fpr_val, or verdict/P/R/F1
    when known_fault_start is given.
    """
    thr_rules = {
        "mean+2std": lambda v: compute_threshold(v, method="mean_std", n_std=2.0),
        "mean+3std": lambda v: compute_threshold(v, method="mean_std", n_std=config.THRESHOLD_N_STD),
        "percentile_99.5": lambda v: compute_threshold(v, method="percentile", percentile=config.THRESHOLD_PERCENTILE),
    }
    rows = []
    for name, r in results.items():
        if "scores_mean" in r:
            aggs = [("mean", r["scores_mean"], np.asarray(r["val_scores_mean"])),
                    ("median", r["scores_median"], np.asarray(r["val_scores_median"]))]
        else:
            aggs = [("as-stored", r["scores"], np.asarray(r["val_scores"]))]
        for agg, full, val in aggs:
            for w in windows:
                for rname, rfn in thr_rules.items():
                    t = float(rfn(val))
                    is_an = full > t
                    alarm = first_confirmed_alarm(is_an, window=w)
                    row = {"model": name, "agg": agg, "window": w, "rule": rname,
                           "threshold": round(t, 5), "alarm": _alarm_str(alarm)}
                    if known_fault_start is not None:
                        v = classify_alarm(alarm, known_fault_start, burn_in_end=burn_in_end)
                        m = score_detection(is_an, known_fault_start,
                                            burn_in_end=burn_in_end, window=w)
                        row.update({"verdict": v["label"],
                                    "delay": (str(v["delay"]) if v["delay"] is not None else None),
                                    "precision": round(m["precision"], 3),
                                    "recall": round(m["recall"], 3),
                                    "f1": round(m["f1"], 3)})
                    else:
                        row["fpr_val"] = round(float(is_an[(full.index > train_end)
                                                           & (full.index <= val_end)].mean()), 4)
                    rows.append(row)
    return pd.DataFrame(rows)


def boundary_sensitivity(models, boundaries: dict[str, pd.Timestamp], feature_cols: list[str],
                         df: pd.DataFrame, val_fraction: float = 0.2,
                         verbose=False, window=None) -> pd.DataFrame:
    """
    Repeat the full procedure (split, scale, fit, threshold) per boundary.
    Uses split_train_val with val_fraction on each healthy period.
    window: continuity window for the confirmed alarm (default None means
    config.CONTINUITY_WINDOW). Threshold always follows config.py defaults.
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
        for row in _score_setup(models, Xtr, Xa_, Xva, df.index, train_end, val_end, window=window):
            rows.append({"boundary_label": label, **row})
    return pd.DataFrame(rows)


def compare_scalers(models, scalers: dict[str, type], df: pd.DataFrame,
                    df_healthy_train: pd.DataFrame, df_healthy_val: pd.DataFrame,
                    train_end, val_end, verbose=False, window=None) -> pd.DataFrame:
    """
    Same procedure under different scalers (classes, instantiated per use).
    window: continuity window for the confirmed alarm (default None means
    config.CONTINUITY_WINDOW). Threshold always follows config.py defaults.
    Returns long table with scaler, model, threshold, alarm, fpr_train, fpr_val.
    """
    rows = []
    for sname, scaler_cls in scalers.items():
        scaler = scaler_cls()
        Xtr = scaler.fit_transform(df_healthy_train)
        Xa_ = scaler.transform(df)
        Xva = scaler.transform(df_healthy_val)
        for row in _score_setup(models, Xtr, Xa_, Xva, df.index, train_end, val_end,
                                label=sname, window=window):
            rows.append({"scaler": sname, **row})
    return pd.DataFrame(rows)


def deep_boundary_sensitivity(boundaries: dict[str, pd.Timestamp], feature_cols: list[str],
                                df: pd.DataFrame, val_fraction: float = 0.2, verbose=False,
                                window=None) -> pd.DataFrame:
    """
    Same as boundary_sensitivity but for deep suite (3 runs, median for LSTM).
    window: continuity window for the confirmed alarm (default None means
    config.CONTINUITY_WINDOW). Threshold always follows config.py defaults.
    """
    from src.preprocessing import split_healthy, split_train_val, scale_features
    rows = []
    for label, b in boundaries.items():
        if verbose:
            print(f"Boundary {label}: training deep suite...", flush=True)
        df_h = split_healthy(df[feature_cols], b)
        df_tr, df_va = split_train_val(df_h, val_fraction=val_fraction)
        Xtr, Xa_, scaler = scale_features(df_tr, df[feature_cols])
        Xva = scaler.transform(df_va)
        train_end, val_end = df_tr.index[-1], df_h.index[-1]
        input_dim = Xtr.shape[1]
        summary, _ = run_deep_suite(Xtr, Xa_, Xva, df.index, train_end, val_end, input_dim, verbose=False, window=window)
        for _, row in summary.iterrows():
            rows.append({"boundary_label": label, "model": row["model"],
                         "threshold": row["threshold"], "alarm": row["first_alarm"],
                         "fpr_train": row["fpr_train"], "fpr_val": row["fpr_val"]})
    return pd.DataFrame(rows)


def deep_compare_scalers(scalers: dict[str, type], df: pd.DataFrame,
                         df_healthy_train: pd.DataFrame, df_healthy_val: pd.DataFrame,
                         train_end, val_end, verbose=False, window=None) -> pd.DataFrame:
    """
    Same as compare_scalers but for deep suite.
    window: continuity window for the confirmed alarm (default None means
    config.CONTINUITY_WINDOW). Threshold always follows config.py defaults.
    """
    rows = []
    for sname, scaler_cls in scalers.items():
        if verbose:
            print(f"Scaler {sname}: training deep suite...", flush=True)
        scaler = scaler_cls()
        Xtr = scaler.fit_transform(df_healthy_train)
        Xa_ = scaler.transform(df)
        Xva = scaler.transform(df_healthy_val)
        input_dim = Xtr.shape[1]
        summary, _ = run_deep_suite(Xtr, Xa_, Xva, df.index, train_end, val_end, input_dim, verbose=False, window=window)
        for _, row in summary.iterrows():
            rows.append({"scaler": sname, "model": row["model"],
                         "threshold": row["threshold"], "alarm": row["first_alarm"],
                         "fpr_train": row["fpr_train"], "fpr_val": row["fpr_val"]})
    return pd.DataFrame(rows)


def format_classical_summary(df: pd.DataFrame):
    display_df = df.copy()
    if "first_alarm" in display_df.columns:
        display_df["first_alarm"] = display_df["first_alarm"].apply(
            lambda x: x.strftime("%Y-%m-%d %H:%M:%S") if isinstance(x, pd.Timestamp) else x
        )
    return display_df.style.format({
        "val_mean": "{:.5f}", "val_std": "{:.5f}",
        "threshold": "{:.5f}", "fpr_train": "{:.4f}", "fpr_val": "{:.4f}"
    })


def compare_joint_24vs36(models, df: pd.DataFrame, df_healthy_train: pd.DataFrame,
                         df_healthy_val: pd.DataFrame, train_end, val_end, verbose=False):
    """
    Joint results on 24 time vs 36 with spectral, plus difference in first alarm.
    Wrapper around run_classical_suite that reuses OLD6 definition.
    Displays three tables and returns (summary_36, summary_24, diff_df).
    """
    from src.preprocessing import scale_features

    OLD6 = ["RMS", "Kurtosis", "CrestFactor", "Peak", "Skewness", "Std"]
    old_cols = [c for c in df.columns if any(c.endswith("_" + f) for f in OLD6)]

    print("Joint results on all 36 with spectral:")
    X_healthy_train, X_all, scaler = scale_features(df_healthy_train, df)
    X_healthy_val = scaler.transform(df_healthy_val)
    summary_36, results_36 = run_classical_suite(
        models, X_healthy_train, X_all, X_healthy_val, df.index, train_end, val_end, verbose=verbose)
    display(format_classical_summary(summary_36))

    Xtr_old, Xa_old, sc_old = scale_features(df_healthy_train[old_cols], df[old_cols])
    Xva_old = sc_old.transform(df_healthy_val[old_cols])
    print("Joint results on old 24 time features:")
    summary_24, results_24 = run_classical_suite(
        models, Xtr_old, Xa_old, Xva_old, df.index, train_end, val_end, verbose=verbose)
    display(format_classical_summary(summary_24))

    print("Difference in first alarm (positive hours = 36 earlier):")
    diff_rows = []
    for name, _, _ in models:
        a24 = results_24[name]["alarm"] if name in results_24 else None
        a36 = results_36[name]["alarm"] if name in results_36 else None
        gain = None
        if a24 is not None and a36 is not None:
            gain = round((a24 - a36).total_seconds() / 3600, 1)
        diff_rows.append({"model": name, "alarm_24": _alarm_str(a24), "alarm_36": _alarm_str(a36),
                          "hours_earlier_36": gain,
                          "fpr_24": round(results_24[name]["fpr_val"], 4) if name in results_24 else None,
                          "fpr_36": round(results_36[name]["fpr_val"], 4) if name in results_36 else None})
    diff_df = pd.DataFrame(diff_rows)
    display(diff_df)
    return summary_36, summary_24, diff_df


def compare_joint_24vs36_full_grid(models, df: pd.DataFrame,
                                   df_healthy_train: pd.DataFrame,
                                   df_healthy_val: pd.DataFrame,
                                   train_end, val_end,
                                   windows: tuple = (10, 15, 20),
                                   verbose=False,
                                   cached_36=None) -> pd.DataFrame:
    """
    Window x threshold-rule grid on joint 24 time vs 36 time+spectral setups.

    Same training as section 8b (sweep_detector_params): fit each model once
    per setup on healthy train, then score every window x rule combo from
    the same scores (no retraining). Column split is identical to
    compare_joint_24vs36: all_24 are the 24 time features (OLD6 set, no
    SpecCentroid/HighFreqRatio/SpecPeak), all_36 is the full frame.
    Scaler is fit per setup on healthy train.

    models: list of (name, fit_fn, score_fn), e.g. the 4 default classical
    detectors. Threshold rules are identical to sweep_detector_params:
    mean+2std, mean+3std (config.THRESHOLD_N_STD), percentile_99.5
    (config.THRESHOLD_PERCENTILE).
    cached_36: optional fitted dict from sweep_detector_params(...,
    return_fitted=True) run on the same all_36 inputs. A cached entry is
    reused only if model name, fit/score functions AND the sha1 data hash
    all match the current all_36 arrays; otherwise the model is refit, so
    a changed boundary or changed hyperparameters can never silently reuse
    stale scores. all_24 has no cache source and is always trained.

    No fault onset is used anywhere here, not even as an argument:
    rows carry alarm + fpr_val only. The table holds ALL combos per setup
    (sorted within each setup by fpr_val, alarm breaks ties, missing
    alarms last) so the notebook can show top-3 per setup and judge
    convincing vs marginal wins, plus pivot like 8b to compare 24 vs 36.
    Columns: setup, model, window, rule, threshold, alarm, fpr_val.
    """
    from src.preprocessing import scale_features

    OLD6 = ["RMS", "Kurtosis", "CrestFactor", "Peak", "Skewness", "Std"]
    old_cols = [c for c in df.columns if any(c.endswith("_" + f) for f in OLD6)]
    setups = {
        "all_24": old_cols,
        "all_36": list(df.columns),
    }
    thr_rules = {
        "mean+2std": lambda v: compute_threshold(v, method="mean_std", n_std=2.0),
        "mean+3std": lambda v: compute_threshold(v, method="mean_std", n_std=config.THRESHOLD_N_STD),
        "percentile_99.5": lambda v: compute_threshold(v, method="percentile", percentile=config.THRESHOLD_PERCENTILE),
    }

    rows = []
    for setup, cols in setups.items():
        if verbose:
            print(f"--- {setup} ---", flush=True)
        Xtr, Xa_, scaler = scale_features(df_healthy_train[cols], df[cols])
        Xva = scaler.transform(df_healthy_val[cols])
        for name, fit_fn, score_fn in models:
            entry = (cached_36 or {}).get(name) if setup == "all_36" else None
            if (entry is not None
                    and entry.get("fit_fn") is fit_fn
                    and entry.get("score_fn") is score_fn
                    and entry.get("data_hash") == _hash_arrays(Xtr, Xa_, Xva)):
                if verbose:
                    print(f"Reusing cached all_36 scores for {name}...", flush=True)
                full, val = entry["full"], entry["val"]
            else:
                if verbose:
                    print(f"Training {name} ({setup})...", flush=True)
                model = fit_fn(Xtr)
                full = pd.Series(score_fn(model, Xa_), index=df.index)
                val = np.asarray(score_fn(model, Xva))
            for w in windows:
                for rname, rfn in thr_rules.items():
                    t = float(rfn(val))
                    is_an = full > t
                    alarm = first_confirmed_alarm(is_an, window=w)
                    fpr_val = float(is_an[(df.index > train_end) & (df.index <= val_end)].mean())
                    rows.append({"setup": setup, "model": name, "window": w,
                                 "rule": rname, "threshold": round(t, 5),
                                 "alarm": _alarm_str(alarm),
                                 "fpr_val": round(fpr_val, 4)})

    out = pd.DataFrame(rows, columns=["setup", "model", "window", "rule",
                                      "threshold", "alarm", "fpr_val"])
    # Sort within each setup by fpr_val, earliest alarm breaks ties (None last).
    out["_alarm_missing"] = out["alarm"].isna()
    out = out.sort_values(["setup", "fpr_val", "_alarm_missing", "alarm", "model", "window", "rule"],
                          ascending=[True, True, True, True, True, True, True]).drop(
        columns=["_alarm_missing"]).reset_index(drop=True)
    return out


def _cached_deep_grid(cached, variants, Xtr, Xa_, Xva, index,
                      train_end, val_end, windows, window_size, verbose=False):
    """
    Re-score cached all_36 deep fits over a window x rule grid (no training).
    Returns a DataFrame with sweep_deep_training_params' no-onset columns,
    or None if any variant lacks a usable entry (label/spec/dims/hash must
    all match) - the caller then refits the whole setup instead of risking
    a partially stale table.
    """
    thr_rules = {
        "mean+2std": lambda v: compute_threshold(v, method="mean_std", n_std=2.0),
        "mean+3std": lambda v: compute_threshold(v, method="mean_std", n_std=config.THRESHOLD_N_STD),
        "percentile_99.5": lambda v: compute_threshold(v, method="percentile", percentile=config.THRESHOLD_PERCENTILE),
    }
    want = {label: (kind, layers, epochs, agg) for label, kind, layers, epochs, agg in variants}
    data_hash = _hash_arrays(Xtr, Xa_, Xva)
    for label, spec in want.items():
        entry = (cached or {}).get(label)
        if (entry is None or (entry.get("kind"), entry.get("layers"),
                              entry.get("epochs"), entry.get("agg")) != spec
                or entry.get("input_dim") != Xtr.shape[1]
                or entry.get("window_size") != window_size
                or entry.get("data_hash") != data_hash):
            return None
    rows = []
    for label in want:
        entry = cached[label]
        if verbose:
            print(f"Reusing cached all_36 scores for {label}...", flush=True)
        scores_series, mean_val, ref_index = entry["full"], entry["val"], entry["ref_index"]
        for w in windows:
            for rname, rfn in thr_rules.items():
                t = float(rfn(mean_val))
                is_an = scores_series > t
                alarm = first_confirmed_alarm(is_an, window=w)
                rows.append({"variant": label, "window": w, "rule": rname,
                             "threshold": round(t, 5), "alarm": _alarm_str(alarm),
                             "fpr_val": round(float(is_an[(ref_index > train_end)
                                                          & (ref_index <= val_end)].mean()), 4)})
    return pd.DataFrame(rows)


def compare_joint_24vs36_full_grid_deep(variants, df: pd.DataFrame,
                                           df_healthy_train: pd.DataFrame,
                                           df_healthy_val: pd.DataFrame,
                                           train_end, val_end,
                                           window_size=10,
                                           windows: tuple = (10, 15, 20),
                                           verbose=False,
                                           cached_36=None) -> pd.DataFrame:
    """
    Window x threshold-rule grid for deep detectors on joint 24 vs 36 setups.

    Same column split as compare_joint_24vs36_full_grid: all_24 are the 24
    time features (OLD6 set, no SpecCentroid/HighFreqRatio/SpecPeak), all_36
    is the full frame. Per setup the columns are scaled and the EXISTING
    sweep_deep_training_params is called over the window x rule grid.
    cached_36: optional fitted dict from sweep_deep_training_params(...,
    return_fitted=True) run on the same all_36 inputs. A cached setup is
    reused only if every variant's label, spec, input_dim, window_size AND
    the sha1 data hash all match; otherwise the whole setup is refit, so a
    changed boundary or variant set can never silently reuse stale scores.
    all_24 has no cache source and is always trained.
    Results get a setup column, are concatenated (pd.concat) and sorted
    within each setup by fpr_val (earliest alarm breaks ties, None last).
    Columns: setup, variant, window, rule, threshold, alarm, fpr_val.

    Selection is onset-blind: no known_fault_start is passed anywhere, rows
    carry alarm + fpr_val only.
    Expensive without cache: len(variants) x N_RUNS x 2 trainings (one fit
    per variant, run and setup). With the full deep_models.deep_training_grid()
    that is 16 variants x 3 runs x 2 setups = 96 trainings; with cached_36
    only the all_24 half (48 trainings) runs.
    """
    from src.preprocessing import scale_features

    OLD6 = ["RMS", "Kurtosis", "CrestFactor", "Peak", "Skewness", "Std"]
    old_cols = [c for c in df.columns if any(c.endswith("_" + f) for f in OLD6)]
    setups = {
        "all_24": old_cols,
        "all_36": list(df.columns),
    }

    frames = []
    for setup, cols in setups.items():
        if verbose:
            print(f"--- {setup} ---", flush=True)
        Xtr, Xa_, scaler = scale_features(df_healthy_train[cols], df[cols])
        Xva = scaler.transform(df_healthy_val[cols])
        sub = None
        if setup == "all_36" and cached_36:
            sub = _cached_deep_grid(cached_36, variants, Xtr, Xa_, Xva, df.index,
                                    train_end, val_end, windows, window_size,
                                    verbose=verbose)
        if sub is None:
            sub = sweep_deep_training_params(
                variants, Xtr, Xa_, Xva, df.index, train_end, val_end,
                input_dim=Xtr.shape[1], window_size=window_size,
                windows=windows, verbose=verbose)
        sub["setup"] = setup
        frames.append(sub)

    out = pd.concat(frames, ignore_index=True)
    out["_alarm_missing"] = out["alarm"].isna()
    out = out.sort_values(["setup", "fpr_val", "_alarm_missing", "alarm",
                           "variant", "window", "rule"],
                          ascending=[True, True, True, True, True, True, True]).drop(
        columns=["_alarm_missing"]).reset_index(drop=True)
    return out


def format_deep_summary(df: pd.DataFrame):
    return format_classical_summary(df)


def run_deep_suite(X_healthy_train, X_all, X_healthy_val, index, train_end, val_end,
                   input_dim, window_size=10, verbose=False, window=None):
    """
    Train 3 dense autoencoders + 1 LSTM autoencoder on healthy train,
    threshold on validation, alarm/FPR on full timeline.
    Handles TF determinism and LSTM median aggregation internally.
    window: continuity window for the confirmed alarm (default None means
    config.CONTINUITY_WINDOW). Threshold always follows config.py defaults.
    Returns (summary DataFrame, results dict) like run_classical_suite.
    """
    import os
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    import tensorflow as tf
    try:
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
    except RuntimeError:
        pass

    from src.deep_models import (
        build_autoencoder, build_lstm_autoencoder, make_windows,
        run_multiple, reconstruction_error, reconstruction_error_lstm,
    )

    results = {}
    architectures = {
        "AE_1layer": [8],
        "AE_2layer": [12, 6],
        "AE_3layer": [16, 8, 4],
    }
    for name, hidden_layers in architectures.items():
        if verbose:
            print(f"Training {name} (3 runs)...", flush=True)
        build_fn = lambda hl=hidden_layers: build_autoencoder(input_dim, hl)
        (mean_full, _), (mean_val, _), all_full, all_val = run_multiple(
            build_fn, X_healthy_train, X_all,
            error_fn=reconstruction_error, X_val=X_healthy_val,
            n_runs=config.N_RUNS, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
            return_runs=True)
        scores_series = pd.Series(mean_full, index=index)
        threshold, alarm, fpr_train, fpr_val, val_mean, val_std = _threshold_alarm_fpr(
            scores_series, mean_val, index, train_end, val_end, window=window)
        results[name] = {"scores": scores_series, "val_scores": np.asarray(mean_val),
                         "scores_mean": pd.Series(mean_full, index=index),
                         "scores_median": pd.Series(np.median(all_full, axis=0), index=index),
                         "val_scores_mean": np.asarray(mean_val),
                         "val_scores_median": np.asarray(np.median(all_val, axis=0)),
                         "val_mean": val_mean, "val_std": val_std,
                         "threshold": float(threshold), "alarm": alarm,
                         "fpr_train": fpr_train, "fpr_val": fpr_val}

    # LSTM on windowed data
    if verbose:
        print("Building LSTM windows...", flush=True)
    X_healthy_windows = make_windows(X_healthy_train, window_size)
    X_all_windows = make_windows(X_all, window_size)
    X_val_windows = make_windows(X_healthy_val, window_size)
    lstm_index = index[window_size - 1:]
    if verbose:
        print("Training LSTM_AE (3 runs, median)...", flush=True)
    build_lstm_fn = lambda: build_lstm_autoencoder(window_size, input_dim, encoding_dim=8)
    (lstm_full, _), (lstm_val, _), lstm_all, lstm_vall = run_multiple(
        build_lstm_fn, X_healthy_windows, X_all_windows,
        error_fn=reconstruction_error_lstm, X_val=X_val_windows, method="median",
        n_runs=config.N_RUNS, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
        return_runs=True)
    lstm_series = pd.Series(lstm_full, index=lstm_index)
    threshold, alarm, fpr_train, fpr_val, val_mean, val_std = _threshold_alarm_fpr(
        lstm_series, lstm_val, lstm_index, train_end, val_end, window=window)
    results["LSTM_AE"] = {"scores": lstm_series, "val_scores": np.asarray(lstm_val),
                          "scores_mean": pd.Series(np.mean(lstm_all, axis=0), index=lstm_index),
                          "scores_median": lstm_series,
                          "val_scores_mean": np.asarray(np.mean(lstm_vall, axis=0)),
                          "val_scores_median": np.asarray(lstm_val),
                          "val_mean": val_mean, "val_std": val_std,
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


def plot_training_curves(X_healthy_train, X_healthy_val, input_dim, window_size=10, spectro_data=None):
    """
    Train one model per architecture with history, plot train vs validation loss.
    If spectro_data is (Xtr_spec, Xva_spec), adds CNN as 5th panel.
    Returns matplotlib figure, no training logic in notebooks.
    """
    import matplotlib.pyplot as plt
    from src.deep_models import train_autoencoder, build_autoencoder, build_lstm_autoencoder, build_cnn_autoencoder, make_windows

    configs = [
        ("AE_1layer", lambda: build_autoencoder(input_dim, [8]), (X_healthy_train, X_healthy_val)),
        ("AE_2layer", lambda: build_autoencoder(input_dim, [12, 6]), (X_healthy_train, X_healthy_val)),
        ("AE_3layer", lambda: build_autoencoder(input_dim, [16, 8, 4]), (X_healthy_train, X_healthy_val)),
        ("LSTM_AE", lambda: build_lstm_autoencoder(window_size, input_dim, encoding_dim=config.ENCODING_DIM),
         (make_windows(X_healthy_train, window_size), make_windows(X_healthy_val, window_size))),
    ]
    if spectro_data is not None:
        Xtr_spec, Xva_spec = spectro_data
        configs.append(("CNN", lambda: build_cnn_autoencoder(64, 64), (Xtr_spec, Xva_spec)))

    n = len(configs)
    cols = 2
    rows = (n + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    axes = axes.flat if n > 1 else [axes]
    for ax, (name, build, data) in zip(axes, configs):
        Xtr_c, Xva_c = data
        m, hist = train_autoencoder(build(), Xtr_c, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, seed=0)
        print(f"{name}: final train loss {hist.history['loss'][-1]:.5f}, val loss {hist.history['val_loss'][-1]:.5f}")
        ax.plot(hist.history["loss"], label="train")
        ax.plot(hist.history["val_loss"], label="validation")
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    for ax in axes[len(configs):]:
        ax.axis("off")
    plt.suptitle("Learning curves per architecture")
    plt.tight_layout()
    plt.close(fig)  # closed so the post-cell flush finds nothing open; the
    # returned figure below still renders exactly once via execute_result.
    # Without this, notebooks show the figure twice (flush display_data +
    # execute_result of the returned fig).
    return fig


def compare_threshold_rules(models, X_healthy_train, X_all, X_healthy_val,
                            index, train_end, val_end, verbose=False):
    """
    Same validation scores, three threshold rules: mean+3std, percentile 99.5, max*1.5.
    Returns long table with model, rule, threshold, alarm, fpr.
    """
    thr_rules = {
        "mean+3std": lambda v: compute_threshold(v, method="mean_std", n_std=config.THRESHOLD_N_STD),
        "percentile_99.5": lambda v: compute_threshold(v, method="percentile", percentile=config.THRESHOLD_PERCENTILE),

    }
    rows = []
    for name, fit_fn, score_fn in models:
        if verbose:
            print(f"Training {name}...", flush=True)
        model = fit_fn(X_healthy_train)
        full = pd.Series(score_fn(model, X_all), index=index)
        val = score_fn(model, X_healthy_val)
        for rname, rfn in thr_rules.items():
            t = rfn(val)
            is_an = full > t
            alarm = first_confirmed_alarm(is_an, window=config.CONTINUITY_WINDOW)
            fpr = float(is_an[(full.index > train_end) & (full.index <= val_end)].mean())
            rows.append({"model": name, "rule": rname, "threshold": round(float(t), 5),
                         "alarm": _alarm_str(alarm), "fpr": round(fpr, 4)})
    return pd.DataFrame(rows)
