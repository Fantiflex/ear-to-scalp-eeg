
"""Channel definitions for the preprocessed Ear-SAAD dataset."""

SCALP_CHANNELS = (
    "O1", "Pz", "P3", "P7", "CP1", "CP5", "M1",
    "Cz", "C3", "T7", "FC1", "FC5", "F3", "F7", "Fp1",
    "O2", "P4", "P8", "CP2", "CP6", "M2",
    "C4", "T8", "FC2", "FC6", "Fz", "F4", "F8", "Fp2",
)

IN_EAR_CHANNELS = (
    "ELA", "ELB", "ELC", "ELE", "ELI", "ELT",
    "ERA", "ERB", "ERC", "ERE", "ERI", "ERT",
)

AROUND_EAR_CHANNELS = (
    "cEL1", "cEL2", "cEL3", "cEL4", "cEL5",
    "cEL6", "cEL7", "cEL8", "cEL9",
    "cER1", "cER2", "cER3", "cER4", "cER5",
    "cER6", "cER7", "cER8", "cER9", "cER10",
)

CHANNEL_GROUPS = {
    "scalp": SCALP_CHANNELS,
    "in_ear": IN_EAR_CHANNELS,
    "around_ear": AROUND_EAR_CHANNELS,
}


def get_channel_indices(channel_names, modality):
    """Return ordered channel indices for a modality."""
    if modality not in CHANNEL_GROUPS:
        raise ValueError(f"Unknown modality: {modality}")

    names = list(channel_names)
    required = CHANNEL_GROUPS[modality]

    missing = [name for name in required if name not in names]

    if missing:
        raise ValueError(
            f"Missing {modality} channels: {missing}"
        )

    return [names.index(name) for name in required]
