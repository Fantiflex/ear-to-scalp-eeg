def make_patient_splits(patient_ids):
    patient_ids = sorted(patient_ids)
    n = len(patient_ids)

    splits = []

    for i in range(n):
        test_patient = patient_ids[i]

        # Two patients immediately following test patient
        val_patients = [
            patient_ids[(i + 1) % n],
            patient_ids[(i + 2) % n]
        ]

        train_patients = [
            p for p in patient_ids
            if p != test_patient and p not in val_patients
        ]

        splits.append({
            "fold": i + 1,
            "train": train_patients,
            "val": val_patients,
            "test": [test_patient]
        })

    return splits