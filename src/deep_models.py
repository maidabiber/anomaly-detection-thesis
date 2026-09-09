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


def train_autoencoder(model, X_train, epochs=50, batch_size=32, validation_split=0.1, seed=42, verbose=0):
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


def run_multiple(build_fn, X_train, X_all, error_fn=reconstruction_error, n_runs=5, seed_start=0,
                  epochs=50, batch_size=32, validation_split=0.1, method="mean"):
    all_errors = []
    for i in range(n_runs):
        seed = seed_start + i
        model = build_fn()
        model, _ = train_autoencoder(model, X_train, epochs=epochs, batch_size=batch_size,
                           validation_split=validation_split, seed=seed)
        errors = error_fn(model, X_all)
        all_errors.append(errors)

    all_errors = np.array(all_errors)
    if method == "median":
        return np.median(all_errors, axis=0), all_errors.std(axis=0)
    return all_errors.mean(axis=0), all_errors.std(axis=0)