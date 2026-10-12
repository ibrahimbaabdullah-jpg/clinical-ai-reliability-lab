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
