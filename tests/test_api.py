"""Exercise dashboard inference, causal replay, and artifact provenance."""
import json

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from clinical_lab import api
from clinical_lab.data import DYNAMIC_COLUMNS


@pytest.fixture(scope="module")
def client():
    # A fixed generated cohort trains once; no API request can upload records.
    import os
    previous = os.environ.pop("CLINICAL_LAB_RUN", None)
    api._load_state.cache_clear()
    with TestClient(api.create_app()) as session:
        yield session
    api._load_state.cache_clear()
    if previous is not None:
        os.environ["CLINICAL_LAB_RUN"] = previous


def _payload(client, **overrides):
    patient = client.get("/api/patients").json()["patients"][0]
    return {"patient_id": patient["id"], "family": "logistic", "hour": 12, **overrides}


def test_dashboard_routes_and_actual_demo_report(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Clinical AI Reliability Lab" in response.text
    assert "script-src 'self'" in response.headers["content-security-policy"]
    for path in ("/static/app.js", "/static/style.css"):
        assert client.get(path).status_code == 200
    status = client.get("/api/status").json()
    assert status["training_source"] == status["replay_source"] == "synthetic"
    assert status["available_models"] == ["logistic"]
    report = client.get("/api/report").json()["report"]
    assert report["metadata"]["source"] == "synthetic"
    assert report["metadata"]["patients"] == 120
    assert report["models"]["logistic"]["external"]["patients"] == 40
    assert len(report["stress_tests"]) == 3


@pytest.mark.parametrize("stress", ["none", "missingness", "delay", "noise"])
def test_replay_uses_only_causal_prefix(client, stress):
    payload = _payload(client, stress=stress)
    current = client.post("/api/replay", json=payload)
    later = client.post("/api/replay", json={**payload, "hour": 18})
    assert current.status_code == later.status_code == 200
    a, b = current.json(), later.json()
    assert a["hours"] == list(range(13))
    # Inference uses float32 features; BLAS kernels can accumulate differently
    # for different batch shapes across CPUs. Feature causality is tested exactly
    # in test_features.py; probabilities allow only float32-scale roundoff.
    np.testing.assert_allclose(a["scores"], b["scores"][:13], rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(a["baseline_scores"], b["baseline_scores"][:13], atol=1e-12)
    assert all(0 <= value <= 1 for value in a["scores"])
    assert a["contribution_scale"] == "uncalibrated_log_odds"
    assert len(a["contributions"]) == 8
    assert a["source"] == "synthetic"


def test_future_measurements_cannot_change_current_score(client):
    payload = _payload(client)
    original = client.post("/api/replay", json=payload).json()["scores"]
    data = api.state()["data"]
    mask = data.patient_id.eq(payload["patient_id"]) & data.hour.gt(payload["hour"])
    backup = data.loc[mask, DYNAMIC_COLUMNS].copy()
    try:
        data.loc[mask, DYNAMIC_COLUMNS] = 99999.0
        changed = client.post("/api/replay", json=payload).json()["scores"]
        np.testing.assert_array_equal(original, changed)
    finally:
        data.loc[mask, DYNAMIC_COLUMNS] = backup


@pytest.mark.parametrize("changes,status", [
    ({"patient_id": "unknown"}, 404),
    ({"patient_id": "../../model.joblib"}, 422),
    ({"family": "gru"}, 404),
    ({"family": "invalid"}, 422),
    ({"severity": "extreme"}, 422),
    ({"stress": "infection"}, 422),
    ({"hour": -1}, 422),
    ({"hour": 500}, 422),
    ({"hour": 3.5}, 422),
    ({"model_path": "/etc/passwd"}, 422),
])
def test_replay_validation_is_clean_4xx(client, changes, status):
    response = client.post("/api/replay", json=_payload(client, **changes))
    assert response.status_code == status
    assert "detail" in response.json()


def test_real_artifact_provenance_keeps_replay_synthetic(client, monkeypatch, tmp_path):
    demo = api.state()
    joblib.dump(demo["models"]["logistic"], tmp_path / "logistic.joblib")
    (tmp_path / "report.json").write_text(json.dumps({"metadata": {"source": "physionet-2019"}}))
    (tmp_path / "training_noise_scales.json").write_text(json.dumps(demo["models"]["logistic"]["noise_scales"]))
    payload = _payload(client, stress="noise")
    monkeypatch.setenv("CLINICAL_LAB_RUN", str(tmp_path))
    api._load_state.cache_clear()
    try:
        status = client.get("/api/status").json()
        assert status["training_source"] == "physionet-2019"
        assert status["replay_source"] == "synthetic"
        replay = client.post("/api/replay", json=payload).json()
        assert replay["out_of_distribution_illustration"] is True
        assert replay["source"] == "synthetic"
    finally:
        monkeypatch.delenv("CLINICAL_LAB_RUN")
        api._load_state.cache_clear()
