"""Controlled measurement availability/value stress tests, never label edits."""
from __future__ import annotations
from collections.abc import Mapping
import numpy as np
import pandas as pd
from .data import DYNAMIC_COLUMNS


def training_scales(training_df: pd.DataFrame) -> dict[str, float]:
    """Fit robust noise units on training data only; freeze before evaluation.

    Uses IQR / 1.349 (a robust Gaussian standard-deviation estimate). Constant or
    entirely absent channels receive scale zero. No test-distribution fitting.
    """
    scales = {}
    for column in DYNAMIC_COLUMNS:
        if column not in training_df:
            scales[column] = 0.0
            continue
        observed = pd.to_numeric(training_df[column], errors="coerce").dropna()
        value = float((observed.quantile(.75)-observed.quantile(.25))/1.349) if len(observed) else 0.0
        if not np.isfinite(value):
            raise ValueError(f"Training column {column} has non-finite scale")
        scales[column] = max(0.0,value)
    return scales


def perturb(df: pd.DataFrame, kind: str, severity: float, seed: int = 2019,
            scales: Mapping[str,float] | None = None) -> pd.DataFrame:
    """Return a corrupted copy of dynamic raw measurements.

    missingness: remove each *observed cell* with independent probability severity.
    delay: move measurement arrivals forward by an integer number of hours inside
      each patient, preserving holes. Shift 3 means a source-hour 0 measurement is
      first visible at hour 3. The hourly data lack true sample timestamps; staleness
      after this transformation is time since delayed availability, not true age.
    noise: add N(0, (severity * frozen_training_scale)^2) to observed values.
      `scales` is mandatory and must come from training data, never test data.

    Static demographics, hospital/subject IDs, hour, ICULOS, labels, and original
    frame are never changed. Perturb before computing causal features. No forward
    fill is done here, so a feature builder can distinguish arrival and absence.
    """
    if not np.isfinite(severity) or severity < 0:
        raise ValueError("Severity must be finite and nonnegative")
    aliases = {"missing":"missingness", "measurement_error":"noise", "delayed_availability":"delay"}
    kind = aliases.get(kind,kind)
    if kind not in {"missingness","delay","noise"}:
        raise ValueError("Kind must be missingness, delay, or noise")
    result = df.copy(deep=True)
    columns = [column for column in DYNAMIC_COLUMNS if column in result]
    if not columns:
        raise ValueError("No dynamic measurement columns found")
    rng = np.random.default_rng(seed)
    # Do not permit nullable/object arrays to turn numerical errors into strings.
    values = result[columns].to_numpy(dtype=float,copy=True)
    if np.isinf(values).any():
        raise ValueError("Raw measurements must not contain infinity")
    if kind == "missingness":
        if severity > 1:
            raise ValueError("Missingness severity is a probability in [0,1]")
        remove = (rng.random(values.shape)<severity) & np.isfinite(values)
        values[remove] = np.nan
        result[columns] = values
    elif kind == "noise":
        if scales is None or any(column not in scales for column in columns):
            raise ValueError("Noise requires training-only scales for every dynamic column")
        frozen = np.asarray([scales[column] for column in columns],dtype=float)
        if not np.isfinite(frozen).all() or (frozen<0).any():
            raise ValueError("Training scales must be finite and nonnegative")
        noise = rng.normal(size=values.shape)*(severity*frozen)
        result[columns] = values+noise  # NaN cells remain missing.
    else:
        if severity != int(severity):
            raise ValueError("Delay severity must be an integer number of hours")
        if not {"patient_id","hospital","hour"} <= set(result):
            raise ValueError("Delay needs hospital, patient_id, and hour")
        delay = int(severity)
        # Sort only within each patient's positional indices; output row/index
        # order is preserved and shifted values can never cross a patient boundary.
        shifted = np.full_like(values,np.nan)
        positions = pd.DataFrame({"hospital":result.hospital.to_numpy(),
            "patient_id":result.patient_id.to_numpy(),"hour":result.hour.to_numpy()})
        if positions.isna().any().any():
            raise ValueError("Delay metadata cannot be missing")
        for _,part in positions.groupby(["hospital","patient_id"],sort=False):
            ordered = part.sort_values("hour")
            hours = ordered.hour.to_numpy()
            if not np.array_equal(hours,np.arange(len(part))):
                raise ValueError("Delay requires contiguous zero-based hourly rows")
            idx = ordered.index.to_numpy()
            if delay == 0:
                shifted[idx] = values[idx]
            elif delay < len(idx):
                shifted[idx[delay:]] = values[idx[:-delay]]
        result[columns] = shifted
    return result
