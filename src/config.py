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

# Reference fault onset per dataset (ground truth for alarm validity).
# Definition: first 20-consecutive exceedance of faulty-bearing RMS above
# healthy-train mean + 3*std. Independent of any ML model (raw signal only),
# same continuity rule (window=20) as model alarms.
#   2nd test, Bearing_1_RMS: healthy mean 0.07720, thr 0.08044 -> 2004-02-16 06:22:39
#   3rd test, Bearing_3_RMS: healthy mean 0.06705, thr 0.07251 -> 2004-04-15 18:22:55
# Rule: confirmed alarm strictly before onset = premature (false), not "early".
# Only alarms at/after onset count as true detections (delay = alarm - onset).
FAULT_ONSET_2ND = "2004-02-16 06:22:39"
FAULT_ONSET_3RD = "2004-04-15 18:22:55"

# Startup transient handling (3rd test only). First 50 rows (2004-03-04, day one)
# are burn-in: excluded from false-positive scoring. An alarm inside this window
# (e.g. IsolationForest 2004-03-04 15:32 in section 9b) is a startup artefact,
# not a detection. 2nd test shows no comparable transient (flat RMS from start).
STARTUP_N_3RD = 50
