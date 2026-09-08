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


def run_multiple(build_fn, X_train, X_all, error_fn=reconstruction_error, n_runs=5, seed_start=0,
                  epochs=50, batch_size=32, validation_split=0.1):
    all_errors = []
    for i in range(n_runs):
        seed = seed_start + i
        model = build_fn()
        train_autoencoder(model, X_train, epochs=epochs, batch_size=batch_size,
                           validation_split=validation_split, seed=seed)
        errors = error_fn(model, X_all)
        all_errors.append(errors)

    all_errors = np.array(all_errors)
    return all_errors.mean(axis=0), all_errors.std(axis=0)