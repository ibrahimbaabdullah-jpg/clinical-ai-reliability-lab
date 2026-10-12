"""Synthetic demonstration cohorts, never a substitute for real validation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from clinical_lab.data import CANONICAL_COLUMNS, DYNAMIC_COLUMNS, VARIABLES, validate_frame


def make_synthetic(n_patients: int = 240, seed: int = 2019) -> pd.DataFrame:
    """Generate two clearly identified synthetic hospitals with hourly records.

    Approximately two thirds of patients belong to A. Sepsis prevalence is
    deliberately enriched to 40% for an efficient software demonstration, not
    estimated from a clinical population. Simulated onset is at least hour 14;
    the supplied label switches on six hours earlier, matching challenge label
    semantics. The onset itself is not placed among the model inputs.
    """
    if not isinstance(n_patients, (int, np.integer)) or n_patients < 6:
        raise ValueError("n_patients must be an integer of at least 6.")
    rng = np.random.default_rng(seed)
    n_a = int(round(n_patients * 2 / 3))
    hospital_sizes = {"A": n_a, "B": n_patients - n_a}
    baselines = np.array([
        82, 97, 36.8, 121, 82, 64, 18, 36, 0, 24, 0.30, 7.40, 40, 97,
        28, 19, 85, 8.8, 102, 0.95, 0.20, 120, 1.4, 2.0, 3.4, 4.1,
        0.70, 0.035, 37, 12.4, 29, 8.5, 300, 230,
    ], dtype=float)
    variation = np.array([
        8, 1, 0.25, 12, 8, 7, 2.5, 4, 2, 2, 0.04, 0.025, 4, 1,
        9, 6, 15, 0.35, 3, 0.20, 0.07, 22, 0.30, 0.18, 0.45, 0.30,
        0.20, 0.015, 3, 1, 4, 2, 55, 55,
    ], dtype=float)
    probability = np.array(
        [0.96, 0.92, 0.48, 0.78, 0.86, 0.71, 0.87, 0.09] + [0.14] * 26
    )
    deterioration = {
        "HR": 24, "Temp": 1.1, "SBP": -17, "MAP": -14, "DBP": -9,
        "Resp": 7, "O2Sat": -2, "Lactate": 2.3, "WBC": 6,
        "Creatinine": 0.40, "Platelets": -55, "BaseExcess": -3,
    }
    frames = []
    serial = 0
    for hospital, count in hospital_sizes.items():
        events = np.zeros(count, dtype=bool)
        events[:max(1, round(count * 0.40))] = True
        rng.shuffle(events)
        for event in events:
            duration = int(rng.integers(24, 61))
            hour = np.arange(duration)
            onset = int(rng.integers(14, duration - 5)) if event else None
            age = float(rng.uniform(20, 88))
            values = baselines + rng.normal(size=34) * variation * 0.8
            values = values[None, :] + rng.normal(size=(duration, 34)) * variation * 0.35
            # Nonsepsis instability and variable event intensity prevent a
            # trivially perfect synthetic classification problem.
            instability = 1 / (1 + np.exp(-(hour - rng.uniform(10, 40)) / 3))
            intensity = float(rng.uniform(0.65, 1.30)) if event else float(rng.uniform(0, 0.40))
            if event:
                instability = 1 / (1 + np.exp(-(hour - (onset - 5)) / 2.5))
            for variable, change in deterioration.items():
                values[:, DYNAMIC_COLUMNS.index(variable)] += change * intensity * instability
            if hospital == "B":
                values[:, DYNAMIC_COLUMNS.index("HR")] += 5
                values[:, DYNAMIC_COLUMNS.index("Glucose")] += 12
                values[:, DYNAMIC_COLUMNS.index("MAP")] -= 3
            values[:, 0:2] = np.maximum(values[:, 0:2], 1)
            values[:, DYNAMIC_COLUMNS.index("O2Sat")] = np.clip(values[:, 1], 75, 100)
            values[:, DYNAMIC_COLUMNS.index("SaO2")] = np.clip(values[:, 13], 75, 100)
            values[:, DYNAMIC_COLUMNS.index("FiO2")] = np.clip(values[:, 10], 0.21, 1)
            for j, variable in enumerate(DYNAMIC_COLUMNS):
                if variable != "BaseExcess":
                    values[:, j] = np.maximum(values[:, j], 0.001)
            availability = probability * (0.82 if hospital == "B" else 1.0)
            observed = rng.random(values.shape) < availability
            values[~observed] = np.nan
            frame = pd.DataFrame(values, columns=DYNAMIC_COLUMNS)
            frame["Age"] = age
            frame["Gender"] = float(rng.integers(0, 2))
            unit = int(rng.integers(0, 2))
            frame["Unit1"], frame["Unit2"] = float(unit == 0), float(unit == 1)
            frame["HospAdmTime"] = -float(rng.uniform(1, 100))
            frame["ICULOS"] = hour + 1
            frame["SepsisLabel"] = (hour >= onset - 6).astype(np.int8) if event else np.zeros(duration, dtype=np.int8)
            frame["patient_id"] = f"{hospital}_synthetic_{serial:05d}"
            frame["hospital"] = hospital
            frame["hour"] = hour.astype(np.int32)
            frames.append(frame)
            serial += 1
    result = pd.concat(frames, ignore_index=True)[CANONICAL_COLUMNS]
    result[VARIABLES] = result[VARIABLES].astype(np.float32)
    result.attrs.update(dataset_kind="synthetic", seed=seed, enriched_sepsis_prevalence=0.40)
    validate_frame(result)
    return result
