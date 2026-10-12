"""Export trusted fitted models as inspectable JSON for pure-JavaScript replay.

This is a direct parameter export, never a distilled or surrogate model. Only
single-output logistic regression, numeric-split binary XGBoost, and this
project's one-layer causal GRU are supported; unknown architectures fail closed.
No Python pickle loader is exposed to a browser or network request.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .data import DYNAMIC_COLUMNS, VARIABLES
from .models import GRUModel, TabularModel, load_model
from .synthetic import make_synthetic

FORMAT = "clinical-lab-portable-v1"
FAMILIES = ("logistic", "xgboost", "gru")


def _numbers(values: Any) -> list:
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError("Portable model weights and preprocessing constants must be finite")
    return values.tolist()


def export_model_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    """Extract the actual trained parameters and frozen calibration/threshold."""
    model = bundle["model"]
    if not isinstance(model, (TabularModel, GRUModel)):
        raise ValueError("Unsupported trained model type")
    preprocessor = model.preprocessor
    width = len(model.feature_columns)
    medians = _numbers(preprocessor.imputer.statistics_)
    if len(medians) != width:
        raise ValueError("Preprocessing and feature schemas disagree")
    scaler = preprocessor.scaler
    result = {
        "format": FORMAT,
        "family": model.family,
        "feature_columns": list(model.feature_columns),
        "ablation": bundle.get("ablation", "full"),
        "threshold": float(bundle["threshold"]),
        "preprocessing": {
            "medians": medians,
            "means": _numbers(scaler.mean_) if scaler is not None else None,
            "scales": _numbers(scaler.scale_) if scaler is not None else None,
            "feature_dtype": "float32",
            "arithmetic": "train-imputation-and-scaling-in-float64-then-float32",
        },
        "calibration": None,
    }
    threshold = result["threshold"]
    if not np.isfinite(threshold) or not 0 <= threshold <= np.nextafter(1., 2.):
        raise ValueError("Invalid frozen threshold")
    calibrator = bundle.get("calibrator")
    if calibrator is not None:
        if getattr(calibrator, "method", None) != "sigmoid_on_logit":
            raise ValueError("Unsupported calibrator")
        coefficients = np.asarray(calibrator.estimator.coef_)
        intercepts = np.asarray(calibrator.estimator.intercept_)
        if coefficients.shape != (1, 1) or intercepts.shape != (1,):
            raise ValueError("Expected binary scalar sigmoid calibration")
        result["calibration"] = {
            "coefficient": float(coefficients[0, 0]), "intercept": float(intercepts[0]),
            "clip": 1e-7,
        }
    if model.family == "logistic":
        estimator = model.estimator
        if estimator.coef_.shape != (1, width) or estimator.classes_.tolist() != [0, 1]:
            raise ValueError("Only binary logistic regression with classes [0,1] is supported")
        result["model"] = {"kind": "logistic", "coefficients": _numbers(estimator.coef_[0]),
                           "intercept": float(estimator.intercept_[0])}
    elif model.family == "xgboost":
        booster = model.estimator.get_booster()
        raw = json.loads(booster.save_raw(raw_format="json"))
        learner = raw["learner"]
        params = learner["learner_model_param"]
        gradient = learner["gradient_booster"]
        if learner["objective"]["name"] != "binary:logistic" or gradient["name"] != "gbtree":
            raise ValueError("Only binary:logistic gbtree XGBoost is portable")
        if int(params.get("num_target", 1)) != 1 or int(params["num_feature"]) != width:
            raise ValueError("Unsupported XGBoost target/feature shape")
        base = json.loads(params["base_score"])
        base = float(base[0] if isinstance(base, list) and len(base) == 1 else base)
        if not 0 < base < 1:
            raise ValueError("XGBoost logistic base_score must lie inside (0,1)")
        trees = []
        for tree in gradient["model"]["trees"]:
            if any(tree["split_type"]) or int(tree["tree_param"].get("size_leaf_vector", 1)) != 1:
                raise ValueError("Categorical/vector-leaf XGBoost trees are not supported")
            trees.append({
                "left": tree["left_children"], "right": tree["right_children"],
                "features": tree["split_indices"], "values": tree["split_conditions"],
                "default_left": tree["default_left"],
            })
        if any(gradient["model"]["tree_info"]):
            raise ValueError("Only one XGBoost output group is supported")
        result["model"] = {"kind": "xgboost", "base_score": base, "trees": trees,
                           "arithmetic": "float32-tree-margin-accumulation"}
    elif model.family == "gru":
        network = model.network
        recurrent = network.gru
        if recurrent.num_layers != 1 or recurrent.bidirectional or not recurrent.batch_first:
            raise ValueError("Only the fitted one-layer causal GRU is portable")
        weights = network.state_dict()
        result["model"] = {
            "kind": "gru", "hidden_size": model.hidden_size,
            "weight_ih": _numbers(weights["gru.weight_ih_l0"].detach().cpu().numpy()),
            "weight_hh": _numbers(weights["gru.weight_hh_l0"].detach().cpu().numpy()),
            "bias_ih": _numbers(weights["gru.bias_ih_l0"].detach().cpu().numpy()),
            "bias_hh": _numbers(weights["gru.bias_hh_l0"].detach().cpu().numpy()),
            "output_weight": _numbers(weights["output.weight"].detach().cpu().numpy()[0]),
            "output_bias": float(weights["output.bias"].detach().cpu().numpy()[0]),
            "gate_order": "reset,update,new", "reset_variant": "pytorch-reset-after-hidden-affine",
            "arithmetic": "float32-with-platform-matmul-rounding-tolerance",
        }
    else:
        raise ValueError(f"Unsupported model family: {model.family}")
    # Refuse NaN/Infinity even if supplied through an unexpected object field.
    json.dumps(result, allow_nan=False)
    return result


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")


def export_browser_run(run: str | Path, output: str | Path, *, synthetic_patients: int = 120,
                       seed: int = 2019) -> dict[str, Any]:
    """Export model parameters, aggregate results, and synthetic-only replay rows.

    ``run`` must be a trusted locally produced artifact directory. Nothing reads
    or exports its cohort, patient manifest, prediction tables, or real records.
    """
    run, output = Path(run), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = json.loads((run / "report.json").read_text())
    models, source_hashes = {}, {}
    for family in FAMILIES:
        path = run / f"{family}.joblib"
        if not path.is_file():
            continue
        bundle = load_model(path)
        if bundle["model"].family != family or bundle.get("family", family) != family:
            raise ValueError(f"Artifact family does not match filename: {family}")
        reported = report.get("models", {}).get(family)
        if not isinstance(reported, dict) or reported.get("threshold") != bundle["threshold"]:
            raise ValueError(f"Artifact threshold/family disagrees with run report: {family}")
        if "parameters" in reported and reported["parameters"] != bundle["model"].params:
            raise ValueError(f"Artifact parameters disagree with run report: {family}")
        models[family] = export_model_bundle(bundle)
        source_hashes[family] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not models:
        raise ValueError("No supported trained model files found")
    _write_json(output / "models.json", models)
    frame = make_synthetic(synthetic_patients, seed=seed)
    patients = []
    for patient_id, part in frame.groupby("patient_id", sort=True):
        # Drop SepsisLabel deliberately: inference never needs the answer label.
        selected = part[[*VARIABLES, "patient_id", "hospital", "hour"]].astype(object)
        records = selected.where(selected.notna(), None).to_dict(orient="records")
        patients.append({"id": patient_id, "hospital": str(part.hospital.iloc[0]),
                         "hours": len(part), "rows": records})
    _write_json(output / "synthetic-patients.json", {"source": "synthetic", "seed": seed, "patients": patients})
    scales = run / "training_noise_scales.json"
    if not scales.is_file():
        raise ValueError("The trained run must include frozen training_noise_scales.json")
    scale_values = json.loads(scales.read_text())
    if set(DYNAMIC_COLUMNS) - set(scale_values) or not all(np.isfinite(value) and value >= 0 for value in scale_values.values()):
        raise ValueError("Incomplete or invalid training-only noise scales")
    _write_json(output / "training-noise-scales.json", scale_values)
    _write_json(output / "report.json", report)
    files = ("models.json", "synthetic-patients.json", "training-noise-scales.json", "report.json")
    manifest = {
        "format": FORMAT, "families": list(models), "model_source_sha256": source_hashes,
        "training_source": report.get("metadata", {}).get("source", "unknown"),
        "replay_source": "synthetic", "replay_patients": len(patients),
        "contains_real_patient_records": False, "contains_pickle": False,
        "feature_definition": "clinical_lab.features.build_features v0.1; causal current-and-past only",
        "attributions": {"logistic": "exact coefficient-times-standardized-feature, before calibration",
                         "xgboost": "unavailable in portable engine; local Python provides native TreeSHAP",
                         "gru": "unavailable"},
        "files_sha256": {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in files},
    }
    _write_json(output / "browser-manifest.json", manifest)
    return manifest
