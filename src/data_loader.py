"""
Low-level helpers for reading raw NASA Bearing sensor files.
Used by features.py to build the actual feature table — this module
does not compute any features itself.
"""
import os
import pandas as pd


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


def list_files_and_columns(data_dir: str, columns: list[str] = None) -> tuple[list[str], list[str]]:
    """
    Lists all files in data_dir and resolves the column/bearing names,
    auto-detecting the number of channels from the first file if
    columns is not given.

    Returns
    -------
    (all_files, columns) : tuple[list[str], list[str]]
    """
    all_paths = sorted(p for p in _find_data_files(data_dir) if _is_sensor_file(p))
    if not all_paths:
        raise ValueError(f"No sensor files found in {data_dir}")

    # return paths relative to data_dir so read_raw_file(data_dir, filename) still works
    all_files = [os.path.relpath(p, data_dir) for p in all_paths]

    first_file = pd.read_csv(all_paths[0], sep='\t', header=None)
    num_channels = first_file.shape[1]

    if columns is None:
        columns = [f"Bearing_{i+1}" for i in range(num_channels)]
    elif len(columns) != num_channels:
        raise ValueError(
            f"Number of column names ({len(columns)}) does not match the "
            f"number of channels in the file ({num_channels})."
        )

    return all_files, columns


def read_raw_file(data_dir: str, filename: str, columns: list[str]) -> pd.DataFrame:
    """Reads a single raw file and returns it as a DataFrame with named columns."""
    path = filename if os.path.isabs(filename) else os.path.join(data_dir, filename)
    file_df = pd.read_csv(path, sep='\t', header=None)
    file_df.columns = columns
    return file_df


def parse_timestamp(filename: str) -> pd.Timestamp:
    """Parses a raw filename (e.g. '2004.02.12.10.32.39') into a timestamp."""
    return pd.to_datetime(os.path.basename(filename), format='%Y.%m.%d.%H.%M.%S')