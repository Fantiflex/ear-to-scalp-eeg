# Ear-to-Scalp EEG Reconstruction

Reconstructing multichannel scalp EEG from sparse **in-ear and around-ear EEG** recordings using deep learning.

This project investigates architectural trade-offs in EEG reconstruction, focusing on temporal and spectral fidelity, cross-subject generalization, robustness to missing electrodes, and computational efficiency.

## Dataset

We use the [Ear-SAAD dataset](https://zenodo.org/records/16536441), containing synchronized recordings from 15 participants:

| Modality | Channels | Role |
|---|---:|---|
| In-ear EEG | 12 | Input |
| Around-ear EEG | 19 | Input |
| Scalp EEG | 29 | Reconstruction target |

The current pipeline uses the preprocessed recordings (20 Hz, 1–9 Hz band-pass), stored locally in `data/raw/preprocessedData/`.

The dataset is not tracked by Git.

## Setup

Requires Python 3.11.

```bash
git clone <REPOSITORY_URL>
cd ear-to-scalp-eeg

python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install -e ".[dev]"
python -m pytest -v
```

Download and extract the preprocessed Ear-SAAD dataset into `data/raw/preprocessedData/`.

## Existing Pipeline

The repository includes:

- `data/channels.py` — EEG channel definitions.
- `data/dataset.py` — synchronized PyTorch dataset with missing-value masks.
- `scripts/inspect_dataset.py` — dataset inspection.
- `scripts/test_dataloader.py` — data loading validation.
- `scripts/visualize_dataset.py` — waveform, spectral, missingness, and electrode visualizations.

Inspect the dataset:

```bash
python scripts/inspect_dataset.py
python scripts/test_dataloader.py
python scripts/visualize_dataset.py --subject 7 --trial 3
```

## Models

| Model | Owner | Status |
|---|---|---|
| MLP | Shared | Planned |
| ? | Hazel | In development |
| Temporal Conformer | Fantine | In development |

Both architectures must support both input modalities and predict the same 29 scalp EEG channels.

### Model Interface

All architectures must implement:

```python
model = Model(
    input_channels=12,  # or 19
    output_channels=29,
)

prediction = model(x, mask=None)
```

Expected tensor shapes:

- Input: `(batch, 12 or 19, time)`
- Optional validity mask: same shape as input
- Output: `(batch, 29, time)`

Each architecture should remain independent of the recording modality, except for the input channel count.

## Experiments

All models will use the same subject-disjoint splits, preprocessing, evaluation windows, and reconstruction targets.

Planned evaluations include:

1. Temporal reconstruction: RMSE and Pearson correlation.
2. Spectral reconstruction: PSD and frequency-band errors.
3. Spatial reconstruction: channel-wise scalp accuracy.
4. Robustness: missing electrodes and channel dropout.
5. Efficiency: parameter count, inference latency, and memory.

Missing scalp target observations will be excluded from loss and metric calculations.

## Project Goal

Determine which architectural inductive biases are most effective for reconstructing spatially distributed scalp EEG from sparse wearable measurements, and how the optimal architecture depends on the input modality.