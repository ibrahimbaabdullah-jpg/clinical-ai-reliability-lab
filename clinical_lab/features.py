"""Causal patient-hour features; no label or hospital information is used."""

from __future__ import annotations

import numpy as np
import pandas as pd

from clinical_lab.data import DYNAMIC_COLUMNS, VARIABLES


ABLATIONS = {"full", "no_trends", "no_missingness", "values_only"}


def build_features(
    df: pd.DataFrame, ablation: str = "full", batch_patients: int | None = 256,
) -> pd.DataFrame:
    """Return float32 predictors in exactly the input row order and index.

    Values are carried forward within patients, never backward. A six-hour
    window includes the current hour and five preceding hours. Its mean uses
    observed measurements only; its slope is the change in carried values
    between the current hour and five hours earlier, divided by five. Slopes
    remain missing until both endpoints are available.

    ``*_missing`` describes the current raw observation. ``*_since`` is hours
    since the most recent measurement, capped at 48; never-measured variables
    also receive 48. Values with no preceding measurement remain NaN so that
    a separately fitted, training-only imputer can handle them.

    For cohorts larger than 20,000 rows, intermediate calculations are batched
    by whole patients and written into one preallocated float32 result. This
    bounds the temporary float64 matrices without truncating patient history.
    Set ``batch_patients=None`` to use the single-cohort reference path.
    """
    if batch_patients is not None and (
        isinstance(batch_patients, bool)
        or not isinstance(batch_patients, (int, np.integer))
        or batch_patients < 1
    ):
        raise ValueError("batch_patients must be a positive integer or None.")
    if len(df) <= 20_000 or batch_patients is None:
        return _build_features(df, ablation)
    # These checks must precede grouping: groupby would otherwise omit missing
    # patient IDs and leave uninitialized rows in the preallocated output.
    if not df.columns.is_unique:
        raise ValueError("Duplicate input columns are not allowed.")
    if "patient_id" not in df:
        raise ValueError("Missing feature inputs: ['patient_id']")
    if df["patient_id"].isna().any():
        raise ValueError("Feature rows require patient identifiers and hours.")
    patient_positions = list(df.groupby("patient_id", sort=False).indices.values())
    values = None
    columns = None
    for start in range(0, len(patient_positions), batch_patients):
        positions = np.concatenate(patient_positions[start:start + batch_patients])
        chunk = _build_features(df.iloc[positions], ablation)
        if values is None:
            columns = chunk.columns
            values = np.empty((len(df), len(columns)), dtype=np.float32)
        values[positions] = chunk.to_numpy(copy=False)
    return pd.DataFrame(values, columns=columns, index=df.index)


def _build_features(df: pd.DataFrame, ablation: str = "full") -> pd.DataFrame:
    """Reference calculation for one complete group of patient trajectories."""
    if ablation not in ABLATIONS:
        raise ValueError(f"Unknown ablation {ablation!r}; choose from {sorted(ABLATIONS)}.")
    required = [*VARIABLES, "patient_id", "hour"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature inputs: {sorted(missing)}")
    if df.empty:
        raise ValueError("Feature input cannot be empty.")
    if not df.columns.is_unique:
        raise ValueError("Duplicate input columns are not allowed.")
    if df[["patient_id", "hour"]].isna().any().any():
        raise ValueError("Feature rows require patient identifiers and hours.")
    if df.duplicated(["patient_id", "hour"]).any():
        raise ValueError("Duplicate patient-hours are not allowed.")
    # A positional index allows even a duplicated caller index to round-trip.
    ordered = df[required].copy()
    ordered["_position"] = np.arange(len(df))
    ordered = ordered.sort_values(["patient_id", "hour"], kind="stable").reset_index(drop=True)
    if not pd.api.types.is_numeric_dtype(ordered["hour"]):
        raise ValueError("hour must be numeric.")
    hours = ordered["hour"].astype(float)
    if not (np.isfinite(hours) & hours.ge(0) & hours.eq(np.floor(hours))).all():
        raise ValueError("hour must contain nonnegative finite integers.")
    groups = ordered.groupby("patient_id", sort=False)
    if not groups["hour"].diff().dropna().eq(1).all():
        raise ValueError("Feature construction requires consecutive hourly rows.")
    numeric = ordered[VARIABLES].astype(float)
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError("Infinite feature inputs are not allowed.")
    patient_keys = ordered["patient_id"]
    carried = numeric.groupby(patient_keys, sort=False).ffill()
    pieces = [carried.add_suffix("_value")]

    if ablation in {"full", "no_missingness"}:
        raw_dynamic = numeric[DYNAMIC_COLUMNS]
        means = raw_dynamic.groupby(patient_keys, sort=False).rolling(
            window=6, min_periods=1
        ).mean().reset_index(level=0, drop=True).reindex(ordered.index)
        previous = carried[DYNAMIC_COLUMNS].groupby(patient_keys, sort=False).shift(5)
        slopes = (carried[DYNAMIC_COLUMNS] - previous) / 5.0
        pieces += [means.add_suffix("_mean6"), slopes.add_suffix("_slope6")]

    if ablation in {"full", "no_trends"}:
        observed = numeric[DYNAMIC_COLUMNS].notna()
        measurement_hours = pd.DataFrame(
            np.broadcast_to(hours.to_numpy()[:, None], observed.shape),
            index=ordered.index, columns=DYNAMIC_COLUMNS,
        ).where(observed)
        last_measurement = measurement_hours.groupby(patient_keys, sort=False).ffill()
        since = last_measurement.rsub(hours, axis=0).fillna(48).clip(lower=0, upper=48)
        pieces += [(~observed).astype(float).add_suffix("_missing"), since.add_suffix("_since")]

    result = pd.concat(pieces, axis=1)
    result["_position"] = ordered["_position"]
    result = result.sort_values("_position", kind="stable").drop(columns="_position")
    result.index = df.index
    return result.astype(np.float32)
