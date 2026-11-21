# Ear-to-Scalp EEG Reconstruction

Reconstructing 29-channel scalp EEG from sparse in-ear and around-ear recordings using deep learning. This project compares neural architectures in terms of reconstruction accuracy, cross-subject generalization, robustness to missing observations, and computational efficiency.

## Dataset

We use the [Ear-SAAD dataset](https://zenodo.org/), comprising synchronized EEG recordings from 15 participants.

| Modality | Channels | Role |
|---|---:|---|
| In-ear EEG | 12 | Input |
| Around-ear EEG | 19 | Input |
| Scalp EEG | 29 | Reconstruction target |

The preprocessed signals are sampled at 20 Hz and segmented into non-overlapping 4-second windows (80 time samples). Missing observations are zero-filled for numerical compatibility, while their original observation masks are preserved.

The dataset is not distributed with this repository.

## Models

- **Temporal Conformer:** temporal convolution and self-attention for EEG reconstruction.
- **Masked Autoencoder:** reconstruction architecture developed as a complementary approach.
- **CNN baseline:** convolutional reference model.

All models are evaluated on the same input modalities, reconstruction targets, and subject-level splits.

## Experimental Protocol

We use **Leave-One-Participant-Out Cross-Validation (LOPO-CV)** across 15 participants. Each fold contains 12 training subjects, 2 validation subjects, and 1 held-out test subject.

Each participant appears exactly once in the test set. Validation is used for model selection and early stopping; test data remain unseen during training.

Reconstruction loss is computed using masked MSE, excluding missing scalp targets. Evaluation includes RMSE, temporal correlation, and spectral reconstruction metrics.

## Setup

Requires Python 3.11.

```bash
git clone https://github.com/Fantiflex/ear-to-scalp-eeg.git
cd ear-to-scalp-eeg

python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Place the preprocessed EAR-SAAD MATLAB recordings under `data/raw/preprocessedData/`.

## Data Preparation

Convert MATLAB recordings into memory-mapped NumPy arrays:

```bash
python scripts/convert_mat_to_numpy.py
```

Converted recordings are stored in `data/processed/numpy/`, preserving trial boundaries, sampling rates, channel indices, and original missing observations. The converted data are checked against the MATLAB loader for numerical equivalence.

## Training

Train the Temporal Conformer on one LOPO-CV fold:

```bash
python scripts/train_conformer.py \
  --modality in_ear \
  --fold 1 \
  --epochs 30 \
  --batch-size 32
```

Use `--modality around_ear` to reconstruct scalp EEG from around-ear electrodes.

Model checkpoints, per-epoch training histories, and validation summaries are saved under `results/conformer/`. Generated datasets and checkpoints are excluded from version control.

## Validation

```bash
python -m pytest -q
python -m ruff check src scripts tests
```

## Project Structure

- `src/eeg_reconstruction/data/` — Data loading, channel selection, and preprocessing.
- `src/eeg_reconstruction/models/` — Reconstruction architectures.
- `src/eeg_reconstruction/training/` — Losses and cross-validation utilities.
- `scripts/` — Data preparation, experiments, and dataset inspection.
- `tests/` — Unit tests and data consistency checks.
- `reports/figures/` — Analysis and reporting figures.

## Research Question

How accurately can distributed scalp EEG activity be reconstructed from sparse wearable ear-EEG recordings, and which architectural choices best support generalization across participants?