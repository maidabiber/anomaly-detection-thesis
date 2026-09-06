import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def split_healthy(df: pd.DataFrame, healthy_boundary: pd.Timestamp) -> pd.DataFrame:
    """Rows before the shared healthy boundary. One boundary for all models."""
    return df[df.index < healthy_boundary]


def scale_features(df_healthy: pd.DataFrame, df_all: pd.DataFrame):
    scaler = MinMaxScaler()
    scaler.fit(df_healthy)
    X_healthy = scaler.transform(df_healthy)
    X_all = scaler.transform(df_all)
    return X_healthy, X_all, scaler


def split_train_val(df_healthy, val_fraction=0.2):
    n_val = int(len(df_healthy) * val_fraction)
    n_train = len(df_healthy) - n_val
    return df_healthy.iloc[:n_train], df_healthy.iloc[n_train:]