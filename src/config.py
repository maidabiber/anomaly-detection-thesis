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

# Reference fault onset: ONLY 2nd test has a documented point.
# Li, W., Qiu, M., Zhu, Z., Jiang, F., Zhou, G., "Fault Diagnosis of Rolling Element
# Bearings with a Spectrum Searching Method," arXiv:1511.03174 (2015):
# record 510 = 2004-02-15 23:22:39, early fault stage via spectral analysis.
# For the 3rd test no equivalent literature source was found, so no onset
# is fixed here — scoring a 3rd-test alarm against an invented onset would
# overclaim. A 3rd-test alarm inside the confirmed healthy period (or the
# startup burn-in) is false by construction, no onset needed.
FAULT_ONSET_2ND = "2004-02-15 23:22:39"

# Operational fault onset for the 3rd test (author-defined: no literature
# source documents one, unlike the 2nd test above). Derivation: Bearing_3 RMS
# over the healthy reference (post-startup rows before HEALTHY_BOUNDARY) has
# mean 0.06750 / std 0.00197, so threshold 0.07341; the first run of 20
# consecutive files above it starts 2004-04-15 15:22:55, jumping to ~0.080+
# sustained through 16.04 (daily mean 0.085) and 17.04 (0.204). All fault-day
# detector alarms (16:32-18:32) follow this physical jump, so delays against
# it discriminate earliness honestly. Operational = transparent assumption,
# not a documented ground truth; the onset-blind verdict column stays.
FAULT_ONSET_3RD_OP = "2004-04-15 15:22:55"

# Startup transient handling (3rd test only). First 50 rows (2004-03-04, day one)
# are burn-in: excluded from false-positive scoring. An alarm inside this window
# (e.g. IsolationForest 2004-03-04 15:32 in section 9b) is a startup artefact,
# not a detection. 2nd test shows no comparable transient (flat RMS from start).
STARTUP_N_3RD = 50
