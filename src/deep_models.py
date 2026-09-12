import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_autoencoder(input_dim, hidden_layers):
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for units in hidden_layers:
        model.add(layers.Dense(units, activation="relu"))
    for units in reversed(hidden_layers[:-1]):
        model.add(layers.Dense(units, activation="relu"))
    model.add(layers.Dense(input_dim, activation="linear"))
    model.compile(optimizer="adam", loss="mse")
    return model


def make_windows(X, window_size):
    n_samples = X.shape[0] - window_size + 1
    n_features = X.shape[1]
    windows = np.zeros((n_samples, window_size, n_features))
    for i in range(n_samples):
        windows[i] = X[i:i + window_size]
    return windows


def build_lstm_autoencoder(window_size, n_features, encoding_dim=8):
    model = keras.Sequential()
    model.add(layers.Input(shape=(window_size, n_features)))
    model.add(layers.LSTM(encoding_dim, activation="relu"))
    model.add(layers.RepeatVector(window_size))
    model.add(layers.LSTM(encoding_dim, activation="relu", return_sequences=True))
    model.add(layers.TimeDistributed(layers.Dense(n_features)))
    model.compile(optimizer="adam", loss="mse")
    return model


def build_cnn_autoencoder(height, width, channels=1):
    """Small convolutional autoencoder for spectrogram frames, sigmoid output for [0, 1] inputs.

    Pilot 2nd test bearing 1 (64x64 log-magnitude, per-frame max normalization):
    reconstructs well (loss ~0.0045) but never reaches 20 consecutive crossings,
    alarm None. Suspected cause: per-frame normalization erases the amplitude
    growth that carries the fault, leaving shape only. Try global normalization next.

    Second pilot, global max from healthy frames only: fault frames reach 3.864
    (sigmoid caps at 1, errors grow), alarm 2004-02-16 07:52:39 with FPR 0.0039.
    Single run, 30 epochs. Average 3 runs before reporting in the thesis.
    """
    model = keras.Sequential()
    model.add(layers.Input(shape=(height, width, channels)))
    model.add(layers.Conv2D(16, 3, activation="relu", padding="same"))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Conv2D(8, 3, activation="relu", padding="same"))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Conv2DTranspose(8, 3, strides=2, activation="relu", padding="same"))
    model.add(layers.Conv2DTranspose(16, 3, strides=2, activation="relu", padding="same"))
    model.add(layers.Conv2D(channels, 3, activation="sigmoid", padding="same"))
    model.compile(optimizer="adam", loss="mse")
    return model


def train_autoencoder(model, X_train, epochs=None, batch_size=None, validation_split=None, seed=42, verbose=0):
    from src.config import EPOCHS, BATCH_SIZE, VALIDATION_SPLIT
    if epochs is None:
        epochs = EPOCHS
    if batch_size is None:
        batch_size = BATCH_SIZE
    if validation_split is None:
        validation_split = VALIDATION_SPLIT
    set_seed(seed)
    history = model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size,
                        validation_split=validation_split, shuffle=True, verbose=verbose)
    return model, history


def reconstruction_error(model, X):
    X_pred = model.predict(X, verbose=0)
    return np.mean((X - X_pred) ** 2, axis=1)


def reconstruction_error_lstm(model, X_windows):
    X_pred = model.predict(X_windows, verbose=0)
    return np.mean((X_windows - X_pred) ** 2, axis=(1, 2))


def reconstruction_error_images(model, X_images):
    X_pred = model.predict(X_images, verbose=0)
    return np.mean((X_images - X_pred) ** 2, axis=(1, 2, 3))


def run_multiple(build_fn, X_train, X_all, error_fn=reconstruction_error, n_runs=None, seed_start=None,
                   epochs=None, batch_size=None, validation_split=None, method="mean",
                   X_val=None, return_runs=False):
    from src.config import EPOCHS, BATCH_SIZE, VALIDATION_SPLIT, N_RUNS, SEED_START
    if n_runs is None:
        n_runs = N_RUNS
    if seed_start is None:
        seed_start = SEED_START
    if epochs is None:
        epochs = EPOCHS
    if batch_size is None:
        batch_size = BATCH_SIZE
    if validation_split is None:
        validation_split = VALIDATION_SPLIT
    """
    Train n_runs models on X_train, return averaged errors on X_all and, if X_val
    is given, on X_val from the SAME models (no retraining). Use X_val to get
    validation errors that share weights with the full errors, so threshold and
    scores come from identical models.
    """
    all_errors = []
    all_val_errors = [] if X_val is not None else None
    for i in range(n_runs):
        seed = seed_start + i
        model = build_fn()
        model, _ = train_autoencoder(model, X_train, epochs=epochs, batch_size=batch_size,
                           validation_split=validation_split, seed=seed)
        all_errors.append(error_fn(model, X_all))
        if X_val is not None:
            all_val_errors.append(error_fn(model, X_val))

    all_errors = np.array(all_errors)
    if method == "median":
        mean_all = np.median(all_errors, axis=0)
        std_all = all_errors.std(axis=0)
    else:
        mean_all = all_errors.mean(axis=0)
        std_all = all_errors.std(axis=0)

    if X_val is None:
        if return_runs:
            return (mean_all, std_all), all_errors
        return mean_all, std_all

    all_val_errors = np.array(all_val_errors)
    if method == "median":
        mean_val = np.median(all_val_errors, axis=0)
        std_val = all_val_errors.std(axis=0)
    else:
        mean_val = all_val_errors.mean(axis=0)
        std_val = all_val_errors.std(axis=0)
    if return_runs:
        return (mean_all, std_all), (mean_val, std_val), all_errors, all_val_errors
    return (mean_all, std_all), (mean_val, std_val)


def deep_training_grid(architectures=None, epochs=(30, 50), include_lstm=True,
                       aggregations=("mean", "median")):
    """Labelled (kind, layers, epochs, agg) variants: architecture x epochs x aggregation.

    Same scoring everywhere downstream; only training differs. Each variant
    costs N_RUNS trainings. Keep the grid small: cost = len(variants) x N_RUNS fits.
    Returns list of (label, kind, layers, epochs, agg).
    """
    if architectures is None:
        architectures = {"AE_1layer": [8], "AE_2layer": [12, 6], "AE_3layer": [16, 8, 4]}
    variants = []
    for arch_name, layers in architectures.items():
        for ep in epochs:
            for agg in aggregations:
                variants.append((f"{arch_name} ep={ep} {agg}", "dense", list(layers), ep, agg))
    if include_lstm:
        for ep in epochs:
            for agg in aggregations:
                variants.append((f"LSTM_AE ep={ep} {agg}", "lstm", None, ep, agg))
    return variants