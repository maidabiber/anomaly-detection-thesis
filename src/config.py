"""
Central hyperparameters for the thesis pipeline.
Change once here, applies everywhere (notebooks + src).
"""
# training
EPOCHS = 30
BATCH_SIZE = 32
VAL_FRACTION = 0.2
VALIDATION_SPLIT = 0.1  # inside train_autoencoder

# threshold
THRESHOLD_METHOD = "mean_std"
THRESHOLD_N_STD = 3.0
THRESHOLD_PERCENTILE = 99.5

# alarm
CONTINUITY_WINDOW = 20

# LSTM / windowing
WINDOW_SIZE = 10
ENCODING_DIM = 8

# deep repetitions
N_RUNS = 3
SEED_START = 0
