"""Guard figure provenance and clinically consequential denominator/NA handling."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("render_figures", ROOT / "scripts/render_figures.py")
figures = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(figures)


@pytest.fixture
def report():
    return json.loads((ROOT / "bundles/pilot-v1/report.json").read_text())


def test_source_intervals_are_not_clamped_and_missing_estimates_are_not_zero(report):
    evaluation = report["models"]["logistic"]["external"]
    evaluation["hourly_metrics"]["average_precision"] = .2
    evaluation["confidence_intervals"]["average_precision"].update(lower=.25, upper=.3)
    row = figures.estimate(evaluation, "average_precision", "hourly_metrics")
    assert (row["estimate"], row["lower"], row["upper"]) == (.2, .25, .3)
    evaluation["hourly_metrics"]["average_precision"] = None
    row = figures.estimate(evaluation, "average_precision", "hourly_metrics")
    assert row["estimate"] is row["lower"] is row["upper"] is None
    assert row["status"] == "undefined"


def test_invalid_density_denominator_and_selective_denominator_are_rejected(report):
    bad = deepcopy(report)
    bad["models"][bad["selected_family"]]["external"]["patient_hours"] += 1
    with pytest.raises(ValueError, match="reconcile"):
        figures.calibration_data(bad)
    bad = deepcopy(report)
    row = bad["selective_prediction"][f"{bad['selected_family']}_external"]["results"][-1]
    row["event_sensitivity_with_withheld_as_no_alert"] = .99
    with pytest.raises(ValueError, match="all eligible events"):
        figures.selective_data(bad)


def test_unknown_interval_methods_are_not_mislabeled(report):
    evaluation = report["models"]["logistic"]["external"]
    evaluation["confidence_intervals"]["average_precision"]["method"] = "unrelated_interval"
    with pytest.raises(ValueError, match="patient-cluster"):
        figures.estimate(evaluation, "average_precision", "hourly_metrics")


def test_rendered_figures_preserve_missingness_and_exact_source_provenance(report, tmp_path):
    report["metadata"]["evidence_status"] = "Explicit test fixture with one undefined metric"
    report["models"]["logistic"]["external"]["hourly_metrics"]["average_precision"] = None
    path, output = tmp_path / "report.json", tmp_path / "figures"
    path.write_text(json.dumps(report))
    manifest = figures.render(path, output)
    assert len(list(output.glob("*.png"))) == len(list(output.glob("*.svg"))) == 4
    assert manifest["source_report_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    for filename, checksum in manifest["files"].items():
        assert hashlib.sha256((output / filename).read_bytes()).hexdigest() == checksum
    data = json.loads((output / "figure-data.json").read_text())
    missing = next(row for row in data["figure_data"]["primary"]
                   if row["family"] == "logistic" and row["split"] == "external" and row["metric"] == "average_precision")
    assert missing["estimate"] is None
    assert "not estimable" in (output / "01_primary_performance.svg").read_text()
    assert "Explicit test fixture" in (output / "01_primary_performance.svg").read_text()
    assert sum(row["count"] for row in data["figure_data"]["calibration"]["bins"]) == data["figure_data"]["calibration"]["patient_hours"]
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in output.glob("*.png"))
    with pytest.raises(ValueError, match="already exists"):
        figures.render(path, output)
