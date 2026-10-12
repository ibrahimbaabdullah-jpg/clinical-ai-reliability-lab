"""Exact regressions for optimizations used by the complete-cohort run.

No clinical outcomes are used to select an optimization or change a protocol.
The reference calculations retain the pre-optimization reduction order.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from clinical_lab.data import (
    CANONICAL_COLUMNS, PSV_COLUMNS, VARIABLES, _validate_psv_values,
    ingest_directory, validate_frame,
)
from clinical_lab.evaluation import (
    PRIMARY_KEYS, _HourlyMetrics, _confidence_intervals, _operational,
    _patient_summaries, _validated, evaluate,
)
from clinical_lab.features import build_features
from clinical_lab.synthetic import make_synthetic


def _legacy_ingest(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        numeric = pd.read_csv(path, sep="|", dtype=str).apply(pd.to_numeric, errors="raise")
        numeric["patient_id"] = "A_" + path.stem
        numeric["hospital"] = "A"
        numeric["hour"] = np.arange(len(numeric), dtype=np.int32)
        validate_frame(numeric)
        numeric["SepsisLabel"] = numeric["SepsisLabel"].astype(np.int8)
        frames.append(numeric)
    return pd.concat(frames, ignore_index=True)[CANONICAL_COLUMNS]


def test_batched_psv_parser_is_exact_across_batch_boundary(tmp_path):
    # More than one 256-file parser batch; full-precision decimal strings and
    # absent values exercise the original numeric parser, not rounded fixtures.
    for patient in range(257):
        rows = []
        for hour in range(4):
            row = ["NaN"] * len(PSV_COLUMNS)
            row[0] = str(0.12345678901234567 + (patient + hour) * 1e-9)
            row[34:40] = ["60", "1", "1", "0", "-42.50", str(hour + 1)]
            row[40] = str(int(patient % 3 == 0 and hour >= 2))
            rows.append("|".join(row))
        text = "|".join(PSV_COLUMNS) + "\n" + "\n".join(rows)
        if patient % 2:
            text += "\n"  # Mixed final-newline presence must not join patients.
        (tmp_path / f"p{patient:06d}.psv").write_text(text)
    expected = _legacy_ingest(sorted(tmp_path.glob("*.psv")))
    actual = ingest_directory(tmp_path, "A")
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


@pytest.mark.parametrize("column,value,row", [
    ("HR", np.inf, 0), ("SepsisLabel", np.nan, 0),
    ("SepsisLabel", 2, 0), ("SepsisLabel", 1, 0),
    ("ICULOS", np.nan, 0), ("ICULOS", 1.5, 0),
    ("ICULOS", 0, 0), ("ICULOS", 9, 2),
])
def test_fast_patient_validation_preserves_rejections(column, value, row):
    numeric = pd.DataFrame(np.nan, index=np.arange(5), columns=PSV_COLUMNS)
    numeric["ICULOS"] = np.arange(1, 6, dtype=float)
    numeric["SepsisLabel"] = 0.0
    numeric.loc[row, column] = value
    canonical = numeric.assign(patient_id="A_test", hospital="A", hour=np.arange(5))
    with pytest.raises(ValueError):
        validate_frame(canonical)
    with pytest.raises(ValueError):
        _validate_psv_values(numeric)


def test_batch_numeric_error_reports_affected_file(tmp_path):
    row = ["NaN"] * len(PSV_COLUMNS)
    row[-2:] = ["1", "0"]
    header = "|".join(PSV_COLUMNS) + "\n"
    (tmp_path / "p000001.psv").write_text(header + "|".join(row) + "\n")
    row[0] = "broken-measurement"
    (tmp_path / "p000002.psv").write_text(header + "|".join(row) + "\n")
    with pytest.raises(ValueError, match="p000002.psv: nonnumeric"):
        ingest_directory(tmp_path, "A")


def _legacy_primary_metrics(y, p, groups, counts):
    """Original hourly bootstrap formulas and reduction order, kept as a guard."""
    y = y.astype(float)
    order = np.argsort(-p, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(p[order]) != 0), len(p) - 1]
    bins = np.minimum((p * 10).astype(int), 9)
    weights = counts[groups].astype(float)
    n = float(weights.sum())
    w, ordered_y = weights[order], y[order]
    tps = np.cumsum(w * ordered_y)[ends]
    fps = np.cumsum(w * (1 - ordered_y))[ends]
    positives, negatives = float(tps[-1]), float(fps[-1])
    auroc = None
    if positives and negatives:
        fpr, tpr = np.r_[0, fps / negatives], np.r_[0, tps / positives]
        auroc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    ap = None
    if positives:
        precision = np.divide(tps, tps + fps, out=np.zeros_like(tps), where=tps + fps > 0)
        ap = float(np.sum(np.diff(np.r_[0, tps / positives]) * precision))
    bins_p = np.bincount(bins, weights=weights * p, minlength=10)
    bins_y = np.bincount(bins, weights=weights * y, minlength=10)
    return dict(auroc=auroc, average_precision=ap,
                brier=float(np.dot(weights, (p - y) ** 2) / n),
                ece=float(np.abs(bins_p - bins_y).sum() / n))


def test_bootstrap_cached_arrays_preserve_exact_draws_and_endpoints():
    frame = make_synthetic(60, seed=51)
    # Ties and endpoint scores are intentional: these catch rank-boundary and
    # calibration-bin changes that continuous random probabilities would miss.
    p = np.random.default_rng(52).choice([0., .1, .2, .5, .9, 1.], size=len(frame))
    frame, p, groups = _validated(frame, p)
    y = frame.SepsisLabel.to_numpy()
    stats = _patient_summaries(frame, p >= .5, groups)
    cache = _HourlyMetrics(y, p, groups)
    rng = np.random.default_rng(53)
    collected = {key: [] for key in PRIMARY_KEYS}
    for _ in range(40):
        counts = np.bincount(rng.integers(0, len(stats), size=len(stats)), minlength=len(stats))
        old = _legacy_primary_metrics(y, p, groups, counts)
        assert cache.metrics(counts, primary_only=True) == old
        old |= _operational(stats, counts)
        for key in PRIMARY_KEYS:
            if old[key] is not None and np.isfinite(old[key]):
                collected[key].append(float(old[key]))
    actual = _confidence_intervals(cache, stats, bootstrap=40, seed=53)
    for key, values in collected.items():
        assert actual[key]["lower"] == (float(np.quantile(values, .025)) if values else None)
        assert actual[key]["upper"] == (float(np.quantile(values, .975)) if values else None)
        assert actual[key]["valid_replicates"] == len(values)


def test_metadata_only_evaluation_preserves_interleaved_patients():
    full = make_synthetic(60, seed=15)
    p = np.random.default_rng(16).choice([0., .25, .5, .75, 1.], size=len(full))
    validated, _, _ = _validated(full, p)
    assert list(validated.columns) == ["patient_id", "hospital", "hour", "SepsisLabel", "Age", "Gender"]
    order = full.sort_values("hour", kind="stable").index.to_numpy()
    expected = evaluate(full, p, .5, bootstrap=10, seed=17)
    actual = evaluate(full.iloc[order], p[order], .5, bootstrap=10, seed=17)
    assert actual == expected


@pytest.mark.parametrize("ablation", ["no_trends", "no_missingness", "values_only"])
def test_large_feature_ablations_match_unbatched_reference(ablation):
    source = make_synthetic(6, seed=19)
    parts = []
    for repeat in range(100):
        part = source.copy()
        part["patient_id"] = str(repeat) + "_" + part["patient_id"]
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=20)
    assert len(frame) > 20_000
    frame.index = np.arange(len(frame)) % 23
    actual = build_features(frame, ablation, batch_patients=31)
    expected = build_features(frame, ablation, batch_patients=None)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
