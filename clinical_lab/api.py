"""Local research dashboard. Replay accepts only bundled synthetic patients.

Set CLINICAL_LAB_RUN to a *trusted local* artifact directory to load a run.
Joblib deserializes executable Python; paths are never accepted over HTTP.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from clinical_lab.data import DYNAMIC_COLUMNS, make_splits
from clinical_lab.evaluation import choose_threshold, evaluate
from clinical_lab.features import build_features
from clinical_lab.models import fit_calibrator, fit_model, load_model
from clinical_lab.synthetic import make_synthetic
from clinical_lab.perturbations import perturb

WEB = Path(__file__).parent / "web"
FAMILIES = ("logistic", "xgboost", "gru")
_state_lock = Lock()


class ReplayRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patient_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    family: Literal["logistic", "xgboost", "gru"] = "logistic"
    hour: int = Field(default=12, ge=0, le=500, strict=True)
    stress: Literal["none", "missingness", "delay", "noise"] = "none"
    severity: Literal["low", "medium", "high"] = "medium"


def _predict(bundle: dict, frame: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    features = build_features(frame, bundle.get("ablation", "full"))
    scores = bundle["model"].predict_proba(features, frame.patient_id)
    if bundle.get("calibrator") is not None:
        scores = bundle["calibrator"].transform(scores)
    return np.asarray(scores), features


def _demo_state() -> dict:
    data = make_synthetic(120, seed=2019)
    splits = make_splits(data, seed=2019)
    frames = {key: data[data.patient_id.isin(ids)].reset_index(drop=True) for key, ids in splits.items()}
    train, tune = frames["train"], frames["tune"]
    model = fit_model("logistic", build_features(train), train.SepsisLabel, train.patient_id,
                      build_features(tune), tune.SepsisLabel, tune.patient_id,
                      config={"C_grid": [0.1]}, seed=2019)
    cal = frames["calibrate"]
    calibrator = fit_calibrator(model.predict_proba(build_features(cal), cal.patient_id), cal.SepsisLabel)
    bundle = {"model": model, "calibrator": calibrator, "threshold": 0.5, "ablation": "full",
              "noise_scales": train[DYNAMIC_COLUMNS].std(ddof=0).fillna(0).to_dict()}
    threshold_frame = frames["threshold"]
    bundle["threshold"] = choose_threshold(threshold_frame, _predict(bundle, threshold_frame)[0])
    results = {key: evaluate(frames[key], _predict(bundle, frames[key])[0], bundle["threshold"], bootstrap=30)
               for key in ("test", "external")}
    stress_tests = []
    for kind in ("missingness", "delay", "noise"):
        changed = _perturb(frames["external"], kind, "medium", bundle["noise_scales"])
        stress_tests.append({"family": "logistic", "split": "external", "kind": kind,
                             "severity": {"missingness": .3, "delay": 3, "noise": .25}[kind],
                             "evaluation": evaluate(changed, _predict(bundle, changed)[0], bundle["threshold"], bootstrap=0)})
    report = {"metadata": {"source": "synthetic", "evidence_status": "synthetic software demonstration",
                           "complete_public_cohort": False, "patients": 120,
                           "patient_hours": len(data), "bootstrap_replicates": 30,
                           "patients_by_hospital": data.groupby("hospital").patient_id.nunique().to_dict(), "seed": 2019},
              "selected_family": "logistic", "stress_tests": stress_tests,
              "disclosure": "Locally trained, enriched synthetic cohort. Software demonstration only; these are not clinical validation results. One fixed logistic configuration; 30 bootstrap replicates for a quick demonstration.",
              "models": {"logistic": {"threshold": bundle["threshold"], **results}}}
    return {"data": data, "models": {"logistic": bundle}, "report": report,
            "source": "synthetic", "demo": True}


@lru_cache(maxsize=1)
def _load_state() -> dict:
    run = os.environ.get("CLINICAL_LAB_RUN")
    if not run:
        return _demo_state()
    directory = Path(run).expanduser().resolve(strict=True)
    report = json.loads((directory / "report.json").read_text())
    models = {}
    scales_path = directory / "training_noise_scales.json"
    scales = json.loads(scales_path.read_text()) if scales_path.is_file() else None
    for family in FAMILIES:
        artifact = directory / f"{family}.joblib"
        if artifact.is_file():
            bundle = load_model(artifact)
            if not isinstance(bundle, dict) or "model" not in bundle or "threshold" not in bundle:
                raise ValueError(f"Invalid {family} artifact")
            bundle["noise_scales"] = scales
            models[family] = bundle
    if not models:
        raise ValueError("The configured run contains no supported model artifacts")
    source = report.get("metadata", {}).get("source", "unknown")
    return {"data": make_synthetic(120, seed=2019), "models": models,
            "report": report, "source": source, "demo": False}


def state() -> dict:
    # Prevent concurrent first requests from training duplicate baselines.
    with _state_lock:
        try:
            return _load_state()
        except Exception as exc:
            import logging
            logging.getLogger(__name__).exception("Research dashboard initialization failed")
            raise HTTPException(503, "Unable to initialize the local run. Check the server log and trusted artifact configuration.") from exc


def _perturb(frame: pd.DataFrame, stress: str, severity: str, scales: dict | None = None) -> pd.DataFrame:
    if stress == "none":
        return frame.copy()
    levels = {"missingness": {"low": .1, "medium": .3, "high": .5},
              "delay": {"low": 1, "medium": 3, "high": 6},
              "noise": {"low": .1, "medium": .25, "high": .5}}
    return perturb(frame, stress, levels[stress][severity], seed=2019, scales=scales)


def _measurements(frame: pd.DataFrame) -> list[dict]:
    names = {"HR": ("Heart rate", "bpm"), "O2Sat": ("Oxygen saturation", "%"),
             "Temp": ("Temperature", "°C"), "SBP": ("Systolic blood pressure", "mmHg"),
             "MAP": ("Mean arterial pressure", "mmHg"), "Resp": ("Respiratory rate", "/min"),
             "Lactate": ("Lactate", "raw units"), "WBC": ("White blood cells", "10³/µL"),
             "Creatinine": ("Creatinine", "mg/dL"), "Platelets": ("Platelets", "10³/µL")}
    current = int(frame.hour.iloc[-1])
    output = []
    for key, (label, unit) in names.items():
        observed = frame.loc[frame[key].notna(), ["hour", key]]
        if observed.empty:
            output.append({"feature": key, "label": label, "unit": unit, "value": None,
                           "age_hours": None, "status": "unmeasured"})
        else:
            age = current - int(observed.hour.iloc[-1])
            output.append({"feature": key, "label": label, "unit": unit,
                           "value": round(float(observed[key].iloc[-1]), 2), "age_hours": age,
                           "status": "observed" if age == 0 else "stale" if age >= 4 else "carried"})
    return output


def create_app() -> FastAPI:
    app = FastAPI(title="Clinical AI Reliability Lab", version="0.1.0",
                  docs_url=None, redoc_url=None,
                  description="Retrospective research platform; synthetic-only patient replay.")

    @app.middleware("http")
    async def security_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(WEB / "index.html")

    @app.get("/api/status")
    def status():
        item = state()
        return {"training_source": item["source"], "replay_source": "synthetic",
                "available_models": list(item["models"]), "demo": item["demo"],
                "intended_use": "Retrospective research and software demonstration only",
                "patient_uploads": False, "clinical_use": False}

    @app.get("/api/patients")
    def patients():
        frame = state()["data"]
        grouped = frame.groupby("patient_id", sort=True)
        return {"source": "synthetic", "patients": [
            {"id": patient, "hospital": str(part.hospital.iloc[0]), "hours": int(part.hour.max()) + 1}
            for patient, part in grouped]}

    @app.get("/api/report")
    def report():
        item = state()
        return {"training_source": item["source"], "demo": item["demo"], "report": item["report"]}

    @app.post("/api/replay")
    def replay(request: ReplayRequest):
        item = state()
        if request.family not in item["models"]:
            raise HTTPException(404, "This model has not been trained in the loaded run")
        full = item["data"].loc[item["data"].patient_id.eq(request.patient_id)].sort_values("hour")
        if full.empty:
            raise HTTPException(404, "Synthetic patient not found")
        if request.hour > int(full.hour.max()):
            raise HTTPException(422, "Requested hour exceeds this synthetic timeline")
        frame = full.loc[full.hour.le(request.hour)].reset_index(drop=True)
        bundle = item["models"][request.family]
        baseline, _ = _predict(bundle, frame)
        scales = bundle.get("noise_scales")
        if request.stress == "noise" and scales is None:
            raise HTTPException(409, "Noise replay requires saved training-only measurement scales")
        changed = _perturb(frame, request.stress, request.severity, scales)
        scores, features = _predict(bundle, changed)
        contributions = bundle["model"].contributions(features, changed.patient_id)
        top = []
        method = None
        if contributions is not None:
            last = contributions.iloc[-1].drop(labels=["bias"], errors="ignore")
            last = last.reindex(last.abs().sort_values(ascending=False).index).head(8)
            top = [{"feature": str(key), "value": float(value)} for key, value in last.items()]
            method = contributions.attrs.get("method", "model_feature_contributions")
        return {"patient_id": request.patient_id, "source": "synthetic", "training_source": item["source"],
                "out_of_distribution_illustration": item["source"] != "synthetic", "family": request.family,
                "hour": request.hour, "threshold": float(bundle["threshold"]), "stress": request.stress,
                "severity": request.severity, "hours": frame.hour.astype(int).tolist(),
                "baseline_scores": baseline.tolist(), "scores": scores.tolist(),
                "measurements": _measurements(changed), "contributions": top,
                "contribution_method": method, "contribution_scale": "uncalibrated_log_odds",
                "warning": "Synthetic illustration, not a clinical assessment. Contributions explain the uncalibrated score, not causes."}

    app.mount("/static", StaticFiles(directory=WEB), name="static")
    return app


app = create_app()
