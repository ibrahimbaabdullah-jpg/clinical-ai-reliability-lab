"""Strict, reproducible ingestion and patient-level cohort partitioning.

The 2019 challenge's ``SepsisLabel`` is used as supplied: its first positive
hour is *not* the sepsis-onset hour. Hospital names and patient identifiers are
metadata, never predictors.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
import pandas as pd


VARIABLES = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets", "Age", "Gender", "Unit1", "Unit2",
    "HospAdmTime", "ICULOS",
]
DYNAMIC_COLUMNS = VARIABLES[:34]
STATIC_COLUMNS = VARIABLES[34:]
PSV_COLUMNS = [*VARIABLES, "SepsisLabel"]
CANONICAL_COLUMNS = [*PSV_COLUMNS, "patient_id", "hospital", "hour"]
SPLIT_FRACTIONS = {
    "train": 0.55,
    "tune": 0.15,
    "calibrate": 0.10,
    "threshold": 0.10,
    "test": 0.10,
}


def validate_frame(df: pd.DataFrame) -> None:
    """Reject malformed canonical input without repairing labels or time.

    ICULOS may start after hour one but must advance by exactly one each row.
    Missing feature values are allowed; missing identifiers, time, and labels
    are not. This intentionally does not reject physiological outliers, since
    they are a subject of the reliability experiments.
    """
    if df.empty:
        raise ValueError("The cohort must contain at least one patient-hour.")
    missing = set(CANONICAL_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing canonical columns: {sorted(missing)}")
    if not df.columns.is_unique:
        raise ValueError("Duplicate column names are not allowed.")
    if df[["patient_id", "hospital", "hour", "SepsisLabel"]].isna().any().any():
        raise ValueError("Patient identifiers, hospital, hour, and labels cannot be missing.")
    if not df["hospital"].isin(["A", "B"]).all():
        raise ValueError("Hospital must be 'A' or 'B'.")
    identities = df[["patient_id", "hospital"]].drop_duplicates()
    if not identities["patient_id"].map(lambda x: isinstance(x, str) and bool(x)).all():
        raise ValueError("patient_id must contain nonempty strings.")
    if not all(pid.startswith(f"{hospital}_") for pid, hospital in
               zip(identities["patient_id"], identities["hospital"])):
        raise ValueError("Patient identifiers must be namespaced with A_ or B_.")
    for column in PSV_COLUMNS + ["hour"]:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"{column} must contain numeric values, not strings.")
        if np.isinf(df[column].to_numpy(dtype=float)).any():
            raise ValueError(f"{column} contains infinite values.")
    if not df["SepsisLabel"].isin([0, 1]).all():
        raise ValueError("SepsisLabel must be observed and binary (0 or 1).")
    if not (df["hour"].ge(0) & df["hour"].eq(np.floor(df["hour"]))).all():
        raise ValueError("hour must be a nonnegative integer.")
    if df.duplicated(["patient_id", "hour"]).any():
        raise ValueError("Duplicate patient-hours are not allowed.")
    if df.groupby("patient_id")["hospital"].nunique().gt(1).any():
        raise ValueError("A patient cannot belong to more than one hospital.")
    # Only temporal metadata need sorting. Copying all clinical measurements
    # here creates a needless full-cohort allocation on every validation pass.
    ordered = df[["patient_id", "hour", "SepsisLabel", "ICULOS"]].sort_values(
        ["patient_id", "hour"], kind="stable"
    )
    groups = ordered.groupby("patient_id", sort=False)
    expected_hour = groups.cumcount().to_numpy()
    if not np.array_equal(ordered["hour"].to_numpy(), expected_hour):
        raise ValueError("Each patient's hour must start at zero and be consecutive.")
    if groups["SepsisLabel"].diff().lt(0).any():
        raise ValueError("SepsisLabel cannot revert from one to zero.")
    if ordered["ICULOS"].isna().any() or not (
        ordered["ICULOS"].ge(1) & ordered["ICULOS"].eq(np.floor(ordered["ICULOS"]))
    ).all():
        raise ValueError("ICULOS must be a positive, observed integer.")
    if not groups["ICULOS"].diff().dropna().eq(1).all():
        raise ValueError("ICULOS must increase by one per patient-hour.")


def _validate_psv_values(numeric: pd.DataFrame | np.ndarray) -> None:
    """Validate a single already-parsed patient without pandas groupby overhead.

    Ingestion separately establishes exact columns, numeric parsing, unique
    subject names, known hospital, and a generated consecutive zero-based hour.
    The checks here cover all remaining ``validate_frame`` invariants while
    retaining the same numeric parser and column dtypes.
    """
    values = numeric.to_numpy(dtype=float) if isinstance(numeric, pd.DataFrame) else numeric
    infinite_columns = np.flatnonzero(np.isinf(values).any(axis=0))
    if len(infinite_columns):
        raise ValueError(f"{PSV_COLUMNS[int(infinite_columns[0])]} contains infinite values.")
    labels = values[:, -1]
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("SepsisLabel must be observed and binary (0 or 1).")
    if np.any(np.diff(labels) < 0):
        raise ValueError("SepsisLabel cannot revert from one to zero.")
    iculos = values[:, -2]
    if not (np.isfinite(iculos) & (iculos >= 1) & (iculos == np.floor(iculos))).all():
        raise ValueError("ICULOS must be a positive, observed integer.")
    if not np.all(np.diff(iculos) == 1):
        raise ValueError("ICULOS must increase by one per patient-hour.")


def _parse_psv_batch(records: list[tuple[Path, str, int]], hospital: str) -> pd.DataFrame:
    """Amortize pandas parsing over bounded patient batches, preserving precision.

    Keeping the original string-to-numeric parser avoids a change to floating
    point rounding. Malformed numeric input takes a diagnostic per-file path;
    ordinary input needs only 41 numeric conversions per batch instead of per
    patient. Record headers and widths have already been checked individually.
    """
    header = "|".join(PSV_COLUMNS) + "\n"
    combined = header + "".join(body if body.endswith("\n") else body + "\n"
                              for _, body, _ in records)
    raw = pd.read_csv(io.StringIO(combined), sep="|", dtype=str)
    try:
        numeric = raw.apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as exc:
        for path, body, _ in records:
            try:
                pd.read_csv(io.StringIO(header + body), sep="|", dtype=str).apply(
                    pd.to_numeric, errors="raise"
                )
            except (ValueError, TypeError) as patient_exc:
                raise ValueError(f"{path.name}: nonnumeric measurement in PSV input.") from patient_exc
        raise ValueError("Nonnumeric measurement in PSV input batch.") from exc
    values = numeric.to_numpy(dtype=float)
    position = 0
    for path, _, length in records:
        try:
            _validate_psv_values(values[position:position + length])
        except ValueError as exc:
            raise ValueError(f"{path.name}: {exc}") from exc
        position += length
    if position != len(numeric):
        raise ValueError("PSV parser row count disagrees with validated record lengths.")
    lengths = [length for _, _, length in records]
    numeric["SepsisLabel"] = numeric["SepsisLabel"].astype(np.int8)
    numeric["patient_id"] = np.repeat([f"{hospital}_{path.stem}" for path, _, _ in records], lengths)
    numeric["hospital"] = hospital
    numeric["hour"] = np.concatenate([np.arange(length, dtype=np.int32) for length in lengths])
    return numeric


def ingest_directory(root: str | Path, hospital: str) -> pd.DataFrame:
    """Read one hospital's challenge PSV files, enforcing the official schema.

    Nested directories are supported. Repeated file stems are rejected rather
    than silently merging patients. Hospital A and B may reuse raw identifiers:
    namespacing keeps their patients separate.
    """
    if hospital not in {"A", "B"}:
        raise ValueError("hospital must be 'A' or 'B'.")
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Not a dataset directory: {root}")
    files = sorted(root.rglob("*.psv"))
    if not files:
        raise ValueError(f"No .psv files found in {root}.")
    subjects: set[str] = set()
    frames: list[pd.DataFrame] = []
    records: list[tuple[Path, str, int]] = []
    for path in files:
        raw_id = path.stem
        if raw_id in subjects:
            raise ValueError(f"Duplicate subject file stem: {raw_id}")
        subjects.add(raw_id)
        text = path.read_text(encoding="utf-8-sig")
        rows = csv.reader(io.StringIO(text), delimiter="|")
        if next(rows, None) != PSV_COLUMNS:
            raise ValueError(f"{path.name}: expected the exact 41-field challenge PSV schema.")
        # pandas can infer a row index for extra fields, or pad a short
        # row with NaN. Both would silently change clinical measurements.
        length = 0
        for line_number, row in enumerate(rows, start=2):
            if len(row) != len(PSV_COLUMNS):
                raise ValueError(
                    f"{path.name}: line {line_number} must contain exactly 41 fields."
                )
            length += 1
        if length == 0:
            raise ValueError(f"{path.name}: patient record contains no hours.")
        records.append((path, text.split("\n", 1)[1], length))
        if len(records) == 256:
            frames.append(_parse_psv_batch(records, hospital))
            records = []
    if records:
        frames.append(_parse_psv_batch(records, hospital))
    result = pd.concat(frames, ignore_index=True)[CANONICAL_COLUMNS]
    result.attrs["dataset_kind"] = "physionet2019"
    return result


def _class_counts(n: int) -> np.ndarray:
    fractions = np.array(list(SPLIT_FRACTIONS.values()), dtype=float)
    exact = n * fractions
    counts = np.floor(exact).astype(int)
    remainder = n - int(counts.sum())
    for position in np.argsort(-(exact - counts), kind="stable")[:remainder]:
        counts[position] += 1
    if np.any(counts == 0):
        raise ValueError("Insufficient patients for all five development partitions.")
    return counts


def make_splits(df: pd.DataFrame, seed: int = 2019) -> dict[str, list[str]]:
    """Stratify A patients into five disjoint sets and freeze all B as external.

    At least ten positive and ten negative A patients are required. Hospital B
    is never used to tune preprocessing, model parameters, calibration, or
    thresholds. Lists are sorted for stable serialization; patient membership
    is determined using the supplied random seed.
    """
    validate_frame(df)
    patients = df.groupby("patient_id", sort=True).agg(
        hospital=("hospital", "first"), sepsis=("SepsisLabel", "max")
    )
    development = patients[patients["hospital"] == "A"]
    external = patients.index[patients["hospital"] == "B"].tolist()
    if not external:
        raise ValueError("A nonempty hospital B cohort is required for external evaluation.")
    rng = np.random.default_rng(seed)
    splits: dict[str, list[str]] = {key: [] for key in SPLIT_FRACTIONS}
    for label in [0, 1]:
        members = development.index[development["sepsis"] == label].to_numpy(dtype=str)
        if len(members) < 10:
            raise ValueError(
                f"Hospital A needs at least 10 patients per class; class {label} has {len(members)}."
            )
        rng.shuffle(members)
        chunks = np.split(members, np.cumsum(_class_counts(len(members)))[:-1])
        for key, chunk in zip(SPLIT_FRACTIONS, chunks):
            splits[key].extend(chunk.tolist())
    for key in splits:
        splits[key].sort()
    splits["external"] = sorted(external)
    return splits


def save_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    """Persist the canonical patient-hour table; raw clinical data stays local."""
    validate_frame(df)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df[CANONICAL_COLUMNS].to_parquet(path, index=False)
    return path


def read_parquet(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    validate_frame(df)
    return df


def cohort_summary(data: pd.DataFrame | str | Path, use_duckdb: bool = False) -> pd.DataFrame:
    """Summarize patients, hours, positive patients, and prevalence by hospital.

    The optional DuckDB path queries Parquet directly without loading the full
    hourly dataset into pandas. SQL parameters bind the file path safely.
    """
    if use_duckdb:
        import duckdb

        with duckdb.connect() as connection:
            if isinstance(data, pd.DataFrame):
                connection.register("cohort", data)
                source, parameters = "cohort", []
            else:
                source, parameters = "read_parquet(?)", [str(data)]
            query = f"""
                WITH patients AS (
                    SELECT hospital, patient_id, COUNT(*) AS hours,
                           MAX(SepsisLabel) AS sepsis
                    FROM {source} GROUP BY hospital, patient_id
                )
                SELECT hospital, COUNT(*) AS patients, SUM(hours) AS patient_hours,
                       SUM(sepsis) AS sepsis_patients, AVG(sepsis) AS prevalence
                FROM patients GROUP BY hospital ORDER BY hospital
            """
            return connection.execute(query, parameters).df()
    df = data if isinstance(data, pd.DataFrame) else read_parquet(data)
    validate_frame(df)
    patients = df.groupby(["hospital", "patient_id"]).agg(
        hours=("hour", "size"), sepsis=("SepsisLabel", "max")
    ).reset_index()
    return patients.groupby("hospital", as_index=False).agg(
        patients=("patient_id", "size"), patient_hours=("hours", "sum"),
        sepsis_patients=("sepsis", "sum"), prevalence=("sepsis", "mean"),
    )
