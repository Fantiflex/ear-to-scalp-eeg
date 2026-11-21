
"""Shared EEG window quality-control rules."""


def is_valid_eeg_window(
    subject_id: int,
    trial_id: int,
    window_idx: int,
) -> bool:
    """Return whether an EEG window passes known QC exclusions.

    Trial IDs are 1-based; window indices are 0-based.

    Subject 4, trial 1, windows 0-149 contain a continuous
    10-minute interval with extreme scalp channel 20 amplitudes.
    """
    return not (subject_id == 4 and trial_id == 1 and window_idx < 150)
