import argparse
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from clinical_lab import cli


def test_end_to_end_frozen_experiment(tmp_path, monkeypatch):
    config = {'seed': 2019, 'families': ['logistic'], 'bootstrap': 3,
              'synthetic_patients': 80, 'stress': True, 'ablations': True,
              'models': {'logistic': {'C_grid': [0.1]}}}
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config))
    output = tmp_path / 'run'
    original = cli.evaluate
    def check_freeze(frame, *args, **kwargs):
        if (frame.hospital == 'B').any():
            assert (output / 'selection.json').exists()
            assert json.loads((output / 'selection.json').read_text())['frozen_before_test_evaluation']
        return original(frame, *args, **kwargs)
    monkeypatch.setattr(cli, 'evaluate', check_freeze)
    args = argparse.Namespace(config=str(path), output=str(output), synthetic=True, data=None)
    report = cli.run(args)
    assert report['metadata']['source'] == 'synthetic'
    assert not report['metadata']['complete_public_cohort']
    assert len(report['stress_tests']) == 18
    assert set(report['ablations']) == {'no_trends', 'no_missingness', 'values_only', 'no_calibration'}
    assert (output / 'logistic.joblib').stat().st_size > 0
    assert (output / 'report.md').is_file()
    json.dumps(report, default=cli.clean_json, allow_nan=False)
    with pytest.raises(ValueError, match='completed report'):
        cli.run(args)


def test_provenance_fails_incomplete_or_changed_download(tmp_path):
    assert not cli.verify_download(tmp_path)['verified']
    path = tmp_path / 'download_manifest.json'
    path.write_text(json.dumps({'status': 'in_progress', 'version': '1.0.0'}))
    with pytest.raises(ValueError, match='incomplete'):
        cli.verify_download(tmp_path)
    (tmp_path / 'patient.psv').write_text('tampered')
    path.write_text(json.dumps({'status': 'complete', 'version': '1.0.0', 'files': []}))
    with pytest.raises(ValueError, match='match the download manifest'):
        cli.verify_download(tmp_path)


def test_report_serializer_handles_query_output_and_nonfinite():
    output = cli.clean_json(pd.DataFrame({'count': [np.int64(2)], 'estimate': [np.nan]}))
    assert output == [{'count': 2, 'estimate': None}]
    json.dumps(output, allow_nan=False)


def test_sqlite_tracking_records_actual_run_metrics_and_artifacts(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow", reason="Install the tracking extra for SQLite integration")
    pytest.importorskip("sqlalchemy")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    original_uri = mlflow.get_tracking_uri()
    config = {"seed": 2019, "families": ["logistic"], "bootstrap": 3,
              "synthetic_patients": 80, "stress": False, "ablations": False,
              "tracking": True, "models": {"logistic": {"C_grid": [0.1]}}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    output = tmp_path / "tracked-run"
    try:
        report = cli.run(argparse.Namespace(config=str(path), output=str(output),
                                            synthetic=True, data=None))
        assert (tmp_path / "mlflow.db").is_file()
        client = mlflow.MlflowClient(tracking_uri="sqlite:///" + str(tmp_path / "mlflow.db"))
        experiment = client.get_experiment_by_name("clinical-ai-reliability-lab")
        runs = client.search_runs([experiment.experiment_id])
        assert len(runs) == 1
        run = runs[0]
        assert run.info.status == "FINISHED"
        assert run.data.params["source"] == "synthetic"
        assert run.data.params["selected_family"] == "logistic"
        assert run.data.params["seed"] == "2019"
        assert run.data.params["data_sha256"] == report["metadata"]["data_sha256"]
        for split in ("test", "external"):
            for metric in ("average_precision", "brier"):
                assert run.data.metrics[f"logistic_{split}_{metric}"] == pytest.approx(
                    report["models"]["logistic"][split]["hourly_metrics"][metric])
        artifacts = {item.path for item in client.list_artifacts(run.info.run_id)}
        assert artifacts == {"report.json", "report.md", "run_manifest.json",
                             "environment.json", "config.yaml", "selection.json"}
        downloaded = client.download_artifacts(run.info.run_id, "report.json")
        from pathlib import Path
        assert json.loads(Path(downloaded).read_text()) == json.loads((output / "report.json").read_text())
    finally:
        mlflow.set_tracking_uri(original_uri)
