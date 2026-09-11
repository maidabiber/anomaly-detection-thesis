import pandas as pd
from src.features import extract_features
from src.preprocessing import split_healthy, scale_features
from src.classical_models import (
    fit_isolation_forest, score_isolation_forest,
    fit_one_class_svm, score_one_class_svm,
    fit_lof, score_lof,
    fit_pca, score_pca,
)
from src.deep_models import (
    build_autoencoder, build_lstm_autoencoder, make_windows,
    run_multiple, reconstruction_error, reconstruction_error_lstm,
)
from src.evaluation import compute_threshold, first_confirmed_alarm, false_positive_rate

df = extract_features('data/raw/2nd_test/')

HEALTHY_BOUNDARY = pd.Timestamp('2004-02-16')
df_healthy = split_healthy(df, HEALTHY_BOUNDARY)
print(f"Healthy samples: {len(df_healthy)} out of {len(df)}")
print(f"Healthy period: {df_healthy.index[0]} to {df_healthy.index[-1]}")

X_healthy, X_all, scaler = scale_features(df_healthy, df)
input_dim = X_healthy.shape[1]


def evaluate(name, scores, index):
    scores_series = pd.Series(scores, index=index)
    mask = index < HEALTHY_BOUNDARY
    healthy_scores = scores[mask]
    threshold = compute_threshold(healthy_scores, method="mean_std", n_std=3.0)
    is_anomaly = scores_series > threshold
    alarm = first_confirmed_alarm(is_anomaly, window=20)
    healthy_end_local = index[mask][-1]
    fpr = false_positive_rate(is_anomaly, healthy_end_local)
    print(f"{name}: threshold={threshold:.5f}  first_alarm={alarm}  FPR={fpr:.4f}")


print("\n--- Classical models ---")
classical = [
    ("IsolationForest", fit_isolation_forest, score_isolation_forest),
    ("OneClassSVM", fit_one_class_svm, score_one_class_svm),
    ("LOF", fit_lof, score_lof),
    ("PCA", fit_pca, score_pca),
]
for name, fit_fn, score_fn in classical:
    model = fit_fn(X_healthy)
    scores = score_fn(model, X_all)
    evaluate(name, scores, df.index)

print("\n--- Deep models (dense autoencoders) ---")
architectures = {
    "AE_1layer": [8],
    "AE_2layer": [12, 6],
    "AE_3layer": [16, 8, 4],
}
for name, hidden_layers in architectures.items():
    build_fn = lambda hl=hidden_layers: build_autoencoder(input_dim, hl)
    mean_errors, std_errors = run_multiple(build_fn, X_healthy, X_all,
                                            error_fn=reconstruction_error,
                                            n_runs=3, epochs=50, batch_size=32)
    evaluate(name, mean_errors, df.index)

print("\n--- LSTM autoencoder ---")
WINDOW_SIZE = 10
X_healthy_windows = make_windows(X_healthy, WINDOW_SIZE)
X_all_windows = make_windows(X_all, WINDOW_SIZE)
lstm_index = df.index[WINDOW_SIZE - 1:]

build_lstm_fn = lambda: build_lstm_autoencoder(WINDOW_SIZE, input_dim, encoding_dim=8)
mean_errors_lstm, std_errors_lstm = run_multiple(
    build_lstm_fn, X_healthy_windows, X_all_windows,
    error_fn=reconstruction_error_lstm,
    n_runs=1, epochs=50, batch_size=32,
)
evaluate("LSTM_AE", mean_errors_lstm, lstm_index)