"""
Module for loading raw NASA Bearing sensor signals and extracting RMS features.
"""
import os
import pandas as pd
import numpy as np


def load_rms_data(data_dir: str, columns: list[str] = None) -> pd.DataFrame:
    """
    Loads all files in data_dir (each file = one measurement in time),
    computes the RMS value per bearing/channel, and returns a table
    indexed by time.

    Parameters
    ----------
    data_dir : str
        Path to the folder with raw files (e.g. "data/raw/2nd_test/").
    columns : list[str], optional
        Column/bearing names. If not given, the number of columns in the
        first file is used to auto-generate names ("Bearing_1", "Bearing_2", ...).

    Returns
    -------
    pd.DataFrame
        Table with a DatetimeIndex ("Time") and one RMS column per channel.
    """
    all_files = sorted(
        path for path in _find_data_files(data_dir) if _is_sensor_file(path)
    )
    if not all_files:
        raise ValueError(f"No sensor files found in {data_dir}")

    first_file = pd.read_csv(all_files[0], sep='\t', header=None)
    num_channels = first_file.shape[1]

    if columns is None:
        columns = [f"Bearing_{i+1}" for i in range(num_channels)]
    elif len(columns) != num_channels:
        raise ValueError(
            f"Number of column names ({len(columns)}) does not match the "
            f"number of channels in the file ({num_channels})."
        )

    rms_rows = []
    for filename in all_files:
        file_df = pd.read_csv(filename, sep='\t', header=None)
        file_df.columns = columns

        rms_row = [np.sqrt(np.mean(file_df[c] ** 2)) for c in columns]

        timestamp = pd.to_datetime(os.path.basename(filename), format='%Y.%m.%d.%H.%M.%S')
        rms_rows.append([timestamp] + rms_row)

    df_rms = pd.DataFrame(rms_rows, columns=['Time'] + columns)
    df_rms.set_index('Time', inplace=True)
    df_rms.sort_index(inplace=True)

    return df_rms


def _find_data_files(data_dir: str) -> list[str]:
    found = []
    for root, _, filenames in os.walk(data_dir):
        for name in filenames:
            found.append(os.path.join(root, name))
    return found


def _is_sensor_file(path: str) -> bool:
    try:
        pd.to_datetime(os.path.basename(path), format='%Y.%m.%d.%H.%M.%S')
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    # Quick test — run "python src/data_loader.py" to check it works
    df = load_rms_data("data/raw/2nd_test/")
    print(df.shape)
    print(df.head())