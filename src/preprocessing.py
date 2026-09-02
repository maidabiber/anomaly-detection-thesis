import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def split_healthy(df: pd.DataFrame, healthy_boundary: pd.Timestamp) -> pd.DataFrame:
    return df[df.index < healthy_boundary]


def scale_features(df_healthy: pd.DataFrame, df_all: pd.DataFrame):
    scaler = MinMaxScaler()
    scaler.fit(df_healthy)
    X_healthy = scaler.transform(df_healthy)
    X_all = scaler.transform(df_all)
    return X_healthy, X_all, scaler