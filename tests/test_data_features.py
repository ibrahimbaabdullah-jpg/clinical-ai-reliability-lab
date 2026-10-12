"""Regression checks for patient separation and causal feature construction."""

import numpy as np
import pandas as pd
import pytest

from clinical_lab.data import (
    CANONICAL_COLUMNS, DYNAMIC_COLUMNS, PSV_COLUMNS, VARIABLES,
    cohort_summary, ingest_directory, make_splits, read_parquet,
    save_parquet, validate_frame,
)
from clinical_lab.features import build_features
from clinical_lab.synthetic import make_synthetic


@pytest.fixture(scope="module")
def cohort():
    return make_synthetic(n_patients=90, seed=8)


def test_official_schema_and_synthetic_labels(cohort):
    assert len(VARIABLES) == 40
    assert len(DYNAMIC_COLUMNS) == 34
    assert list(cohort.columns) == CANONICAL_COLUMNS
    assert cohort["patient_id"].str.contains("synthetic").all()
    assert cohort.attrs["dataset_kind"] == "synthetic"
    validate_frame(cohort)


def test_psv_ingestion_and_namespacing(tmp_path, cohort):
    patient = cohort[cohort.patient_id == cohort.patient_id.iloc[0]][PSV_COLUMNS]
    patient.to_csv(tmp_path / "p000001.psv", sep="|", index=False)
    loaded = ingest_directory(tmp_path, "A")
    assert loaded.patient_id.unique().tolist() == ["A_p000001"]
    assert loaded.hour.tolist() == list(range(len(patient)))
    np.testing.assert_allclose(loaded[PSV_COLUMNS], patient, equal_nan=True, rtol=1e-6)
    altered = patient.rename(columns={"HR": "wrong"})
    altered.to_csv(tmp_path / "p000002.psv", sep="|", index=False)
    with pytest.raises(ValueError, match="41-field"):
        ingest_directory(tmp_path, "A")


@pytest.mark.parametrize("failure", ["numeric", "reversed_label", "time_gap", "missing_label"])
def test_invalid_psv_rejected(tmp_path, cohort, failure):
    patient = cohort[cohort.patient_id == cohort.patient_id.iloc[0]][PSV_COLUMNS].copy().reset_index(drop=True)
    if failure == "numeric":
        patient["HR"] = patient["HR"].astype(object)
        patient.loc[0, "HR"] = "bad-value"
    elif failure == "reversed_label":
        patient["SepsisLabel"] = 0
        patient.loc[0, "SepsisLabel"] = 1
    elif failure == "time_gap":
        patient.loc[1, "ICULOS"] += 2
    else:
        patient["SepsisLabel"] = patient["SepsisLabel"].astype(float)
        patient.loc[0, "SepsisLabel"] = np.nan
    patient.to_csv(tmp_path / "p000001.psv", sep="|", index=False)
    with pytest.raises(ValueError):
        ingest_directory(tmp_path, "A")


def test_duplicate_subject_file_rejected(tmp_path, cohort):
    patient = cohort[cohort.patient_id == cohort.patient_id.iloc[0]][PSV_COLUMNS]
    (tmp_path / "nested").mkdir()
    patient.to_csv(tmp_path / "p000001.psv", sep="|", index=False)
    patient.to_csv(tmp_path / "nested" / "p000001.psv", sep="|", index=False)
    with pytest.raises(ValueError, match="Duplicate subject"):
        ingest_directory(tmp_path, "A")


@pytest.mark.parametrize("fields", [40, 42])
def test_psv_row_width_is_strict(tmp_path, fields):
    path = tmp_path / "p000001.psv"
    path.write_text("|".join(PSV_COLUMNS) + "\n" + "|".join(["1"] * fields) + "\n")
    with pytest.raises(ValueError, match="exactly 41 fields"):
        ingest_directory(tmp_path, "A")


def test_features_are_prefix_invariant(cohort):
    patient = cohort[cohort.patient_id == cohort.patient_id.iloc[0]].copy()
    before = build_features(patient.iloc[:12])
    after = build_features(patient)
    pd.testing.assert_frame_equal(before, after.iloc[:12])
    # Arbitrarily extreme later observations still cannot alter earlier hours.
    patient.loc[patient.index[12:], DYNAMIC_COLUMNS] = 9999.0
    pd.testing.assert_frame_equal(before, build_features(patient).iloc[:12])


def test_missing_measurements_and_causal_trend(cohort):
    patient = cohort[cohort.patient_id == cohort.patient_id.iloc[0]].copy()
    patient["Lactate"] = np.nan
    patient["HR"] = np.nan
    patient.loc[patient.index[2], "HR"] = 80
    patient.loc[patient.index[7], "HR"] = 100
    result = build_features(patient)
    assert result["Lactate_value"].isna().all()
    assert result["Lactate_missing"].eq(1).all()
    assert result["Lactate_since"].eq(48).all()
    assert result["HR_value"].iloc[:2].isna().all()
    assert result["HR_value"].iloc[3] == 80
    assert result["HR_since"].iloc[4] == 2
    assert result["HR_slope6"].iloc[7] == 4
    assert result["HR_mean6"].iloc[7] == 90


def test_features_do_not_use_other_patients(cohort):
    selected = cohort[cohort.patient_id.isin(cohort.patient_id.unique()[:2])].copy()
    selected["Lactate"] = np.nan
    first = selected.patient_id.iloc[0]
    selected.loc[selected.patient_id == first, "Lactate"] = 9
    shuffled = selected.sample(frac=1, random_state=2)
    result = build_features(shuffled)
    assert result.index.equals(shuffled.index)
    assert result.loc[shuffled.patient_id != first, "Lactate_value"].isna().all()
    assert all(dtype == np.float32 for dtype in result.dtypes)
    assert not {"patient_id", "hospital", "SepsisLabel", "hour"}.intersection(result.columns)


def test_large_batched_features_equal_reference_with_duplicate_index(cohort):
    copies = []
    for number in range(6):
        copy = cohort.copy()
        copy["patient_id"] = str(number) + "_" + copy["patient_id"]
        copies.append(copy)
    large = pd.concat(copies, ignore_index=True).sample(frac=1, random_state=5)
    assert len(large) > 20_000
    large.index = pd.Index(np.arange(len(large)) % 17, name="duplicate_index")
    batched = build_features(large, batch_patients=17)
    reference = build_features(large, batch_patients=None)
    pd.testing.assert_frame_equal(batched, reference)


@pytest.mark.parametrize("ablation,columns", [
    ("full", 176), ("no_trends", 108), ("no_missingness", 108), ("values_only", 40),
])
def test_feature_ablations(cohort, ablation, columns):
    assert build_features(cohort.iloc[:60], ablation=ablation).shape == (60, columns)


def test_patient_splits_disjoint_stratified_and_reproducible(cohort):
    splits = make_splits(cohort, seed=4)
    assert splits == make_splits(cohort.sample(frac=1, random_state=10), seed=4)
    assigned = [patient for patients in splits.values() for patient in patients]
    assert len(assigned) == len(set(assigned)) == cohort.patient_id.nunique()
    assert set(assigned) == set(cohort.patient_id)
    assert all(patient.startswith("B_") for patient in splits["external"])
    labels = cohort.groupby("patient_id")["SepsisLabel"].max()
    for key in ["train", "tune", "calibrate", "threshold", "test"]:
        assert all(patient.startswith("A_") for patient in splits[key])
        assert set(labels.loc[splits[key]]) == {0, 1}


def test_small_cohort_is_rejected_without_leaky_fallback():
    with pytest.raises(ValueError, match="at least 10"):
        make_splits(make_synthetic(n_patients=12))


def test_parquet_roundtrip_and_summary(tmp_path, cohort):
    path = save_parquet(cohort, tmp_path / "cohort.parquet")
    loaded = read_parquet(path)
    pd.testing.assert_frame_equal(loaded, cohort)
    summary = cohort_summary(path)
    assert summary.patients.sum() == 90
    assert summary.patient_hours.sum() == len(cohort)
    pytest.importorskip("duckdb")
    other = cohort_summary(path, use_duckdb=True)
    pd.testing.assert_frame_equal(summary, other, check_dtype=False)
