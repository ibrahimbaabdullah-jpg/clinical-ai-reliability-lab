"""Numerical parity of real fitted Python models and browser inference."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from clinical_lab.data import VARIABLES, make_splits
from clinical_lab.export_models import export_browser_run, export_model_bundle
from clinical_lab.features import build_features
from clinical_lab.models import fit_calibrator, fit_model, load_model, save_model
from clinical_lab.synthetic import make_synthetic

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "clinical_lab" / "web" / "inference.js"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js is needed for portable-engine parity")
NODE_SCRIPT = """
import fs from 'node:fs';
const source=fs.readFileSync(process.argv[1],'utf8');
const {predict,buildFeatures,preprocess}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
const input=JSON.parse(fs.readFileSync(0,'utf8'));
const output=input.cases.map(rows=>{const features=buildFeatures(rows,input.bundle.ablation);return {...predict(input.bundle,rows),features,preprocessed:preprocess(input.bundle,features)};});
console.log(JSON.stringify(output));
"""


def _rows(frame):
    # Nulls are the only JSON missingness encoding. No labels enter JavaScript.
    selected = frame[[*VARIABLES, "patient_id", "hospital", "hour"]].astype(object)
    return selected.where(selected.notna(), None).to_dict(orient="records")


def _javascript(bundle, cases):
    result = subprocess.run([NODE, "--input-type=module", "-e", NODE_SCRIPT, str(ENGINE)],
                            input=json.dumps({"bundle": bundle, "cases": [_rows(frame) for frame in cases]}, allow_nan=False),
                            text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


@pytest.fixture(scope="module")
def replay_cases():
    frame = make_synthetic(30, seed=2019)
    patients = [part.reset_index(drop=True) for _, part in frame.groupby("patient_id")]
    return [patients[0].iloc[:1], patients[0].iloc[:6], patients[0].iloc[:17], patients[0],
            patients[5], patients[-1], pd.concat([patients[0], patients[-1]], ignore_index=True)]


@pytest.fixture(scope="module")
def trained_bundles(tmp_path_factory):
    # Lightweight CI coverage remains real model training, never fabricated weights.
    frame = make_synthetic(80, seed=31)
    split = make_splits(frame, seed=31)
    frames = {key: frame[frame.patient_id.isin(ids)].reset_index(drop=True) for key, ids in split.items()}
    features = {key: build_features(part) for key, part in frames.items()}
    config = {"logistic": {"C_grid": [0.1]}, "xgboost": {"param_grid": [{"max_depth": 2, "n_estimators": 4}]},
              "gru": {"hidden_size_grid": [4], "epochs": 1, "batch_size": 16}}
    output = tmp_path_factory.mktemp("portable-trained-models")
    bundles = {}
    for family in ("logistic", "xgboost", "gru"):
        if family == "gru":
            pytest.importorskip("torch")
        model = fit_model(family, features["train"], frames["train"].SepsisLabel, frames["train"].patient_id,
                          features["tune"], frames["tune"].SepsisLabel, frames["tune"].patient_id, config, seed=31)
        cal = model.predict_proba(features["calibrate"], frames["calibrate"].patient_id)
        bundle = {"model": model, "calibrator": fit_calibrator(cal, frames["calibrate"].SepsisLabel),
                  "family": family, "ablation": "full", "threshold": 0.05}
        path = save_model(bundle, output / f"{family}.joblib")
        bundles[family] = load_model(path)
    return bundles


def _assert_parity(bundle, cases):
    exported = export_model_bundle(bundle)
    results = _javascript(exported, cases)
    errors = []
    for frame, result in zip(cases, results):
        features = build_features(frame)
        js_features = pd.DataFrame(result["features"])[features.columns].to_numpy(dtype=float)
        np.testing.assert_allclose(js_features, features.to_numpy(), rtol=0, atol=0, equal_nan=True)
        processed = bundle["model"].preprocessor.transform(features)
        np.testing.assert_allclose(result["preprocessed"], processed, rtol=0, atol=0)
        raw = bundle["model"].predict_proba(features, frame.patient_id)
        calibrated = bundle["calibrator"].transform(raw)
        # Float32 PyTorch/XGBoost kernels have implementation-specific summation;
        # tolerances are small numerical errors, not surrogate-model agreement.
        np.testing.assert_allclose(result["raw_scores"], raw, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(result["scores"], calibrated, rtol=2e-5, atol=2e-6)
        np.testing.assert_array_equal(np.asarray(result["scores"]) >= bundle["threshold"],
                                      calibrated >= bundle["threshold"])
        errors.append(float(np.max(np.abs(np.asarray(result["scores"]) - calibrated))))
        if bundle["model"].family == "logistic":
            exact = bundle["model"].contributions(features).iloc[-1]
            for contribution in result["contributions"]:
                assert contribution["value"] == pytest.approx(exact[contribution["feature"]], rel=1e-12, abs=1e-12)
        else:
            assert result["contributions"] == []
            assert result["contribution_unavailable_reason"]
    # Last inference of a longer prefix cannot alter already-emitted scores.
    full = np.asarray(results[3]["scores"])
    for index in range(3):
        np.testing.assert_array_equal(results[index]["scores"], full[:len(cases[index])])
    return max(errors)


@pytest.mark.parametrize("family", ["logistic", "xgboost", "gru"])
def test_browser_matches_actual_trained_saved_models(trained_bundles, replay_cases, family):
    _assert_parity(trained_bundles[family], replay_cases)


@pytest.mark.parametrize("family", ["logistic", "xgboost", "gru"])
def test_browser_matches_actual_saved_physionet_pilot(replay_cases, family):
    artifact = ROOT / "bundles" / "pilot-v1" / f"{family}.joblib"
    if not artifact.is_file():
        pytest.skip("Optional local PhysioNet pilot artifacts are not redistributed in source checkout")
    error = _assert_parity(load_model(artifact), replay_cases)
    print(f"{family} pilot maximum calibrated-probability error: {error:.9g}")


def test_export_contains_only_weights_aggregate_results_and_synthetic_rows(tmp_path, trained_bundles):
    run, output = tmp_path / "run", tmp_path / "public"
    run.mkdir()
    for family, bundle in trained_bundles.items():
        save_model(bundle, run / f"{family}.joblib")
    (run / "training_noise_scales.json").write_text(json.dumps({name: 1 for name in VARIABLES[:34]}))
    (run / "report.json").write_text(json.dumps({"metadata": {"source": "synthetic"}, "models": {family: {"threshold": bundle["threshold"]} for family, bundle in trained_bundles.items()}}))
    # These deliberately sensitive-looking files must never be read/exported.
    (run / "cohort.parquet").write_text("DO_NOT_EXPORT_RAW_PATIENTS")
    (run / "split_manifest.json").write_text("DO_NOT_EXPORT_PATIENT_IDS")
    manifest = export_browser_run(run, output, synthetic_patients=6)
    assert manifest["contains_real_patient_records"] is False
    assert manifest["contains_pickle"] is False
    assert set(manifest["families"]) == {"logistic", "xgboost", "gru"}
    for path in output.iterdir():
        text = path.read_text()
        assert "DO_NOT_EXPORT" not in text
        assert "SepsisLabel" not in text
    patients = json.loads((output / "synthetic-patients.json").read_text())
    assert len(patients["patients"]) == 6
    assert any(row["Lactate"] is None for p in patients["patients"] for row in p["rows"])


@pytest.mark.parametrize("run_name", ["pilot-v1", "full-v1"])
def test_all_actual_exported_replay_patients_keep_frozen_alert_decisions(tmp_path, run_name):
    """Exercise the exact 120-patient public fixture, not merely selected prefixes."""
    run = ROOT / "bundles" / run_name
    if not all((run / f"{family}.joblib").is_file() for family in ("logistic", "xgboost", "gru")):
        pytest.skip("Local pilot artifacts are optional outside the trained workspace")
    export_browser_run(run, tmp_path, synthetic_patients=120)
    script = """
import fs from 'node:fs';
const source=fs.readFileSync(process.argv[1],'utf8');
const {predict}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
const root=process.argv[2];
const models=JSON.parse(fs.readFileSync(root+'/models.json','utf8'));
const patients=JSON.parse(fs.readFileSync(root+'/synthetic-patients.json','utf8')).patients;
const output=Object.fromEntries(Object.entries(models).map(([family,bundle])=>[family,patients.flatMap(patient=>predict(bundle,patient.rows).scores)]));
console.log(JSON.stringify(output));
"""
    result = subprocess.run([NODE, "--input-type=module", "-e", script, str(ENGINE), str(tmp_path)],
                            text=True, capture_output=True, check=True)
    scores = json.loads(result.stdout)
    patients = json.loads((tmp_path / "synthetic-patients.json").read_text())["patients"]
    frame = pd.DataFrame([row for patient in patients for row in patient["rows"]])
    features = build_features(frame)
    for family, browser_probabilities in scores.items():
        bundle = load_model(run / f"{family}.joblib")
        probabilities = bundle["calibrator"].transform(bundle["model"].predict_proba(features, frame.patient_id))
        np.testing.assert_allclose(browser_probabilities, probabilities, rtol=2e-5, atol=2e-6)
        np.testing.assert_array_equal(np.asarray(browser_probabilities) >= bundle["threshold"],
                                      probabilities >= bundle["threshold"])
