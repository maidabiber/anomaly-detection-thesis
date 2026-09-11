"""
High-level experiment runners: same procedure for every model, results as tables.
Notebook cells call these and display the output, no training logic in notebooks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import src.config as config
from src.evaluation import compute_threshold, first_confirmed_alarm, false_positive_rate


def _alarm_str(alarm) -> str | None:
    return alarm.strftime("%Y-%m-%d %H:%M:%S") if alarm is not None else None


def _threshold_alarm_fpr(scores_series: pd.Series, val_scores, index, train_end, val_end):
    """Shared threshold (mean+3std), alarm (window) and FPRs. One place, used everywhere."""
    val_mean = float(np.mean(val_scores))
    val_std = float(np.std(val_scores))
    threshold = compute_threshold(val_scores, method="mean_std", n_std=config.THRESHOLD_N_STD)
    is_anomaly = scores_series > threshold
    alarm = first_confirmed_alarm(is_anomaly, window=config.CONTINUITY_WINDOW)
    fpr_train = float(is_anomaly[index <= train_end].mean())
    fpr_val = float(is_anomaly[(index > train_end) & (index <= val_end)].mean())
    return threshold, alarm, fpr_train, fpr_val, val_mean, val_std


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
        scores_series = pd.Series(full_scores, index=index)
        threshold, alarm, fpr_train, fpr_val, val_mean, val_std = _threshold_alarm_fpr(
            scores_series, val_scores, index, train_end, val_end)
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
                   label: str | None = None) -> list[dict]:
    rows = []
    for name, fit_fn, score_fn in models:
        if label is not None:
            print(f"Training {name} ({label})...", flush=True)
        model = fit_fn(Xtr)
        full = pd.Series(score_fn(model, Xa_), index=index)
        val = score_fn(model, Xva)
        threshold, alarm, fpr_train, fpr_val, _, _ = _threshold_alarm_fpr(
            full, val, index, train_end, val_end)
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


def compare_feature_setups(models, df: pd.DataFrame, df_healthy_train: pd.DataFrame,
                           df_healthy_val: pd.DataFrame, train_end, val_end,
                           feature_types: list[str] | None = None, verbose=False) -> pd.DataFrame:
    """
    Joint (all_24 vs all_36) plus per-feature setups in one table.
    Uses OLD6 (24 time features) and all 36 (time+spectral) as two joints,
    then one entry per feature type. Wrapper around compare_column_setups.
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
                                 train_end, val_end, verbose=verbose)


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


def format_deep_summary(df: pd.DataFrame):
    return format_classical_summary(df)


def run_deep_suite(X_healthy_train, X_all, X_healthy_val, index, train_end, val_end,
                   input_dim, window_size=10, verbose=False):
    """
    Train 3 dense autoencoders + 1 LSTM autoencoder on healthy train,
    threshold on validation, alarm/FPR on full timeline.
    Handles TF determinism and LSTM median aggregation internally.
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
        (mean_full, _), (mean_val, _) = run_multiple(
            build_fn, X_healthy_train, X_all,
            error_fn=reconstruction_error, X_val=X_healthy_val,
            n_runs=config.N_RUNS, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE)
        scores_series = pd.Series(mean_full, index=index)
        threshold, alarm, fpr_train, fpr_val, val_mean, val_std = _threshold_alarm_fpr(
            scores_series, mean_val, index, train_end, val_end)
        results[name] = {"scores": scores_series, "val_mean": val_mean, "val_std": val_std,
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
    (lstm_full, _), (lstm_val, _) = run_multiple(
        build_lstm_fn, X_healthy_windows, X_all_windows,
        error_fn=reconstruction_error_lstm, X_val=X_val_windows, method="median",
        n_runs=config.N_RUNS, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE)
    lstm_series = pd.Series(lstm_full, index=lstm_index)
    threshold, alarm, fpr_train, fpr_val, val_mean, val_std = _threshold_alarm_fpr(
        lstm_series, lstm_val, lstm_index, train_end, val_end)
    results["LSTM_AE"] = {"scores": lstm_series, "val_mean": val_mean, "val_std": val_std,
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
    plt.show()
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
        "max_x1.5": lambda v: float(np.max(v) * 1.5),
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
