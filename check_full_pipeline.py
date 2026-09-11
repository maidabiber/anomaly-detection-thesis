import pandas as pd
from src.features import extract_features
from src.preprocessing import split_healthy, scale_features
from src.classical_models import (
    fit_isolation_forest, score_isolation_forest,
    fit_one_class_svm, score_one_class_svm,
    fit_lof, score_lof,
    fit_pca, score_pca,
)
from src.evaluation import compute_threshold, first_confirmed_alarm, false_positive_rate

df = extract_features('data/raw/2nd_test/')

HEALTHY_BOUNDARY = pd.Timestamp('2004-02-16')
df_healthy = split_healthy(df, HEALTHY_BOUNDARY)
print(f"Healthy samples: {len(df_healthy)} out of {len(df)}")
print(f"Healthy period: {df_healthy.index[0]} to {df_healthy.index[-1]}")

X_healthy, X_all, scaler = scale_features(df_healthy, df)
healthy_end = df_healthy.index[-1]
healthy_mask = df.index < HEALTHY_BOUNDARY

models = [
    ('IsolationForest', fit_isolation_forest, score_isolation_forest),
    ('OneClassSVM', fit_one_class_svm, score_one_class_svm),
    ('LOF', fit_lof, score_lof),
    ('PCA', fit_pca, score_pca),
]

for name, fit_fn, score_fn in models:
    model = fit_fn(X_healthy)
    scores = score_fn(model, X_all)
    scores_series = pd.Series(scores, index=df.index)

    healthy_scores = scores[healthy_mask]
    threshold = compute_threshold(healthy_scores, method='mean_std', n_std=3.0)

    is_anomaly = scores_series > threshold
    alarm = first_confirmed_alarm(is_anomaly, window=20)
    fpr = false_positive_rate(is_anomaly, healthy_end)

    print(f"{name}: threshold={threshold:.4f}  first_alarm={alarm}  FPR={fpr:.4f}")