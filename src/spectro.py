"""
Spectrogram extraction per bearing for CNN autoencoders.
One file = one frame. Cached to data/processed to avoid recompute.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy.signal import stft

from src.data_loader import list_files_and_columns, read_raw_file, parse_timestamp


def _frame_logmag(data_dir: str, filename: str, columns: list[str], bearing: str,
                  nperseg: int) -> np.ndarray:
    file_df = read_raw_file(data_dir, filename, columns)
    signal = file_df[bearing].values - np.mean(file_df[bearing].values)
    _, _, spec = stft(signal, nperseg=nperseg)
    return np.log1p(np.abs(spec))


def compute_healthy_max(data_dir: str, bearing: str, healthy_end: pd.Timestamp,
                        nperseg: int = 256, columns: list[str] = None) -> float:
    """Max log-magnitude over healthy files only. Global reference, no leakage."""
    all_files, columns = list_files_and_columns(data_dir, columns)
    peak = 0.0
    for filename in all_files:
        if parse_timestamp(filename) < healthy_end:
            peak = max(peak, float(_frame_logmag(data_dir, filename, columns, bearing, nperseg).max()))
    return peak


def extract_spectrograms(data_dir: str, bearing: str = "Bearing_1",
                         nperseg: int = 256, target_size: tuple = (64, 64),
                         columns: list[str] = None,
                         global_ref: float | None = None) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """
    STFT log-magnitude spectrograms, resized to target_size, one per file.

    Normalization: per-frame max when global_ref is None, otherwise every frame
    is divided by global_ref (use compute_healthy_max so fault frames may exceed 1).

    Returns
    -------
    (X, index) : X has shape (n_files, target_h, target_w), index holds timestamps.
    """
    from scipy.ndimage import zoom as _zoom

    all_files, columns = list_files_and_columns(data_dir, columns)
    frames = []
    stamps = []
    for filename in all_files:
        mag = _frame_logmag(data_dir, filename, columns, bearing, nperseg)
        scale = global_ref if global_ref else (mag.max() if mag.max() > 0 else 1.0)
        mag = mag / scale
        zy, zx = target_size[0] / mag.shape[0], target_size[1] / mag.shape[1]
        small = _zoom(mag, (zy, zx), order=1)
        frames.append(small.astype(np.float32))
        stamps.append(parse_timestamp(filename))

    order = np.argsort(stamps)
    return np.array(frames)[order], pd.DatetimeIndex(np.array(stamps)[order])


def cache_path(data_dir: str, bearing: str, norm: str = "perframe") -> str:
    tag = os.path.basename(os.path.normpath(data_dir)).replace(" ", "_")
    return os.path.join("data", "processed", f"spectro_{tag}_{bearing}_{norm}.npz")


def save_spectrograms(path: str, X: np.ndarray, index: pd.DatetimeIndex) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, X=X, index=index.values)


def load_spectrograms(path: str) -> tuple[np.ndarray, pd.DatetimeIndex]:
    pack = np.load(path)
    return pack["X"], pd.DatetimeIndex(pack["index"])
