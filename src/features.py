"""
Module for extracting vibration features per bearing/channel.
Time domain: RMS, kurtosis, crest factor, peak, skewness, standard deviation.
Frequency domain (rfft, normalized frequencies, no sampling rate assumed):
spectral centroid, high band power ratio, spectral peak.
Reuses file-reading helpers from data_loader.py instead of re-implementing them.
"""
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from src.data_loader import list_files_and_columns, read_raw_file, parse_timestamp


def extract_features(data_dir: str, columns: list[str] = None) -> pd.DataFrame:
    """
    Loads all files in data_dir and computes a set of time-domain features
    per bearing/channel for each file (each file = one point in time).

    Parameters
    ----------
    data_dir : str
        Path to the folder with raw files (e.g. "data/raw/2nd_test/").
    columns : list[str], optional
        Column/bearing names. Auto-detected from the first file if not given.

    Returns
    -------
    pd.DataFrame
        Table with a DatetimeIndex ("Time") and, for each bearing, columns:
        {bearing}_RMS, {bearing}_Kurtosis, {bearing}_CrestFactor,
        {bearing}_Peak, {bearing}_Skewness, {bearing}_Std,
        {bearing}_SpecCentroid, {bearing}_HighFreqRatio, {bearing}_SpecPeak
    """
    all_files, columns = list_files_and_columns(data_dir, columns)

    rows = []
    for filename in all_files:
        file_df = read_raw_file(data_dir, filename, columns)

        row = {}
        for c in columns:
            signal = file_df[c].values
            rms = np.sqrt(np.mean(signal ** 2))
            peak = np.max(np.abs(signal))

            row[f"{c}_RMS"] = rms
            row[f"{c}_Kurtosis"] = kurtosis(signal, fisher=True)  # 0 = normal distribution
            row[f"{c}_CrestFactor"] = peak / rms if rms != 0 else np.nan
            row[f"{c}_Peak"] = peak
            row[f"{c}_Skewness"] = skew(signal)
            row[f"{c}_Std"] = np.std(signal)

            spectrum = np.abs(np.fft.rfft(signal)) ** 2
            total_power = np.sum(spectrum)
            freqs = np.linspace(0, 1, len(spectrum))
            if total_power > 0:
                row[f"{c}_SpecCentroid"] = float(np.sum(freqs * spectrum) / total_power)
                row[f"{c}_HighFreqRatio"] = float(np.sum(spectrum[freqs > 0.5]) / total_power)
                row[f"{c}_SpecPeak"] = float(np.max(spectrum) / total_power)
            else:
                row[f"{c}_SpecCentroid"] = 0.0
                row[f"{c}_HighFreqRatio"] = 0.0
                row[f"{c}_SpecPeak"] = 0.0

        row["Time"] = parse_timestamp(filename)
        rows.append(row)

    df_features = pd.DataFrame(rows)
    df_features.set_index("Time", inplace=True)
    df_features.sort_index(inplace=True)

    return df_features


if __name__ == "__main__":
    # Quick test — run "python src/features.py" to check it works
    df = extract_features("data/raw/2nd_test/")
    print(df.shape)
    print(df.columns.tolist())
    print(df.head())