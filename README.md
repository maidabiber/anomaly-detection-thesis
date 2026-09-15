# Anomaly Detection on NASA IMS Bearing Data (Thesis)

Classical detectors (IsolationForest, OneClassSVM, LOF, PCA), dense/LSTM
autoencoders and a CNN on spectrograms for early bearing-fault detection.
Compares a shared winning configuration across the 2nd and 3rd IMS tests
(transfer check), with window x threshold-rule sweeps throughout.

## Data

Raw files are too large for git: download the NASA IMS bearing dataset
(Kaggle, "NASA Bearing Dataset") and unpack so that
`data/raw/1st_test/`, `data/raw/2nd_test/`, `data/raw/3rd_test/` hold the
timestamped sensor files (see `docs/ims_bearing_readme.pdf` for the
original documentation).
Note: this copy of the 3rd test holds 6,324 files through 2004-04-18,
while the IMS document describes 4,448 files ending 2004-04-04 - the
analysis uses the files on disk; the discrepancy is stated in the thesis.

`data/processed/` holds generated spectrogram caches (`.npz`, rebuilt
automatically) and `results/cache/` holds trained-score caches - both are
git-ignored and regenerate on first run (see caching note below).

## Install

Reference environment: CPython 3.9, Windows 10/11.

```bash
pip install -r requirements.txt
```

Pinned versions reproduce the thesis numbers. Newer scikit-learn or
TensorFlow builds may shift single-feature alarms by minutes (see the
reproducibility notes in the notebooks).

## Run

Open the notebooks in order (run top to bottom; restart the kernel first
for final numbers):

1. `notebooks/analyze_2nd_test.ipynb` - full pipeline on the 2nd test
   (documented fault onset `2004-02-15 23:22:39`, Li et al. 2015).
2. `notebooks/analyze_3rd_test_v2.ipynb` - current 3rd-test analysis
   (startup cut, operational onset `2004-04-15 15:22:55`, see
   `src/config.py`). `analyze_3rd_test.ipynb` is the superseded draft.

`check_all_models.py` / `check_full_pipeline.py` are quick smoke checks.

### Caching (no retraining for tables)

Expensive trainings write `results/cache/<test>/...pkl` with a sidecar
`.meta.json` (config snapshot + data hash + setup). Cache hits validate
the metadata, so a changed boundary or hyperparameter never silently
reuses stale scores. Final-comparison and validity cells load from
memory first, disk cache second - after one full pass only those cells
need re-running.

## Layout

- `src/config.py` - all shared hyperparameters, onsets, startup handling
- `src/features.py`, `src/data_loader.py` - raw files to feature table
- `src/classical_models.py`, `src/deep_models.py` - detectors
- `src/evaluation.py` - thresholds, confirmed alarms, onset scoring
  (`classify_alarm`, `detection_delay`, `precision_recall_f1`)
- `src/experiments.py` - sweep runners used by every notebook cell
- `src/spectro.py` - STFT spectrograms for the CNN
- `src/cache.py` - disk cache with staleness guards
