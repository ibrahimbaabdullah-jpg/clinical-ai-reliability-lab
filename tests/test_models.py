"""Checks for clinically consequential leakage and model/artifact behavior."""
import importlib.util

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

from clinical_lab.models import fit_calibrator, fit_model, load_model, save_model


def example_data():
    rng = np.random.default_rng(23)
    train = pd.DataFrame({"signal": rng.normal(size=80), "sometimes_missing": rng.normal(size=80),
                          "never_measured": np.full(80, np.nan)})
    tune = pd.DataFrame({"signal": rng.normal(size=32), "sometimes_missing": rng.normal(size=32),
                         "never_measured": np.full(32, np.nan)})
    train.loc[::4, "sometimes_missing"] = np.nan
    tune.loc[::3, "sometimes_missing"] = np.nan
    y_train = (train.signal.to_numpy() + rng.normal(scale=0.3, size=80) > 0).astype(int)
    y_tune = (tune.signal.to_numpy() + rng.normal(scale=0.3, size=32) > 0).astype(int)
    ids_train = np.repeat([f"train_{i}" for i in range(20)], 4)
    ids_tune = np.repeat([f"tune_{i}" for i in range(8)], 4)
    return train, y_train, ids_train, tune, y_tune, ids_tune


def test_logistic_preprocessing_ignores_tuning_distribution_and_keeps_empty_columns():
    data = list(example_data())
    expected_median = np.nanmedian(data[0].sometimes_missing)
    data[3]["sometimes_missing"] = 10_000.0
    model = fit_model("logistic", *data, config={"C_grid": [0.1, 1.0]})
    assert model.preprocessor.imputer.statistics_[1] == pytest.approx(expected_median)
    assert model.preprocessor.transform(data[3]).shape[1] == 3
    probabilities = model.predict_proba(data[3], data[5])
    assert probabilities.shape == (len(data[3]),)
    assert np.isfinite(probabilities).all()
    assert len(model.params["tuning_scores"]) == 2
    np.testing.assert_allclose(probabilities, model.predict_proba(data[3][list(reversed(data[3].columns))], data[5]))


def test_logistic_contributions_reconstruct_only_uncalibrated_probability(tmp_path):
    train, y_train, train_ids, tune, y_tune, tune_ids = example_data()
    model = fit_model("logistic", train, y_train, train_ids, tune, y_tune, tune_ids,
                      config={"C_grid": [1.0]})
    contributions = model.contributions(tune, tune_ids)
    np.testing.assert_allclose(expit(contributions.sum(axis=1)), model.predict_proba(tune, tune_ids), atol=1e-7)
    assert contributions.attrs["scale"] == "uncalibrated_log_odds"
    saved = save_model(model, tmp_path / "logistic.joblib")
    loaded = load_model(saved)
    np.testing.assert_array_equal(loaded.predict_proba(tune, tune_ids), model.predict_proba(tune, tune_ids))


def test_patient_overlap_is_rejected_before_training():
    data = list(example_data())
    data[5] = data[5].astype(object)
    data[5][0] = data[2][0]
    with pytest.raises(ValueError, match="disjoint"):
        fit_model("logistic", *data)


def test_invalid_labels_and_mismatched_prediction_schema_are_rejected():
    data = list(example_data())
    original_labels = data[4].copy()
    data[4][:] = 0
    with pytest.raises(ValueError, match="both positive and negative"):
        fit_model("logistic", *data)
    data[4] = original_labels
    model = fit_model("logistic", *data, config={"C_grid": [1]})
    with pytest.raises(ValueError, match="schema"):
        model.predict_proba(data[3].drop(columns="signal"), data[5])
    with pytest.raises(ValueError, match="infinity"):
        model.predict_proba(data[3].assign(signal=np.inf), data[5])


def test_calibration_is_fitted_on_heldout_probabilities_and_roundtrips(tmp_path):
    probabilities = np.linspace(0.01, 0.99, 200)
    rng = np.random.default_rng(5)
    labels = rng.binomial(1, expit(0.4 * logit(probabilities) - 1.0))
    calibrator = fit_calibrator(probabilities, labels)
    calibrated = calibrator.transform(probabilities)
    assert calibrated.shape == probabilities.shape
    assert ((calibrated > 0) & (calibrated < 1)).all()
    assert ((calibrated - labels) ** 2).mean() < ((probabilities - labels) ** 2).mean()
    artifact = save_model(calibrator, tmp_path / "calibrator.joblib")
    np.testing.assert_array_equal(load_model(artifact).transform(probabilities), calibrated)
    with pytest.raises(ValueError, match="both positive and negative"):
        fit_calibrator(probabilities, np.zeros(200))
    with pytest.raises(ValueError, match="between 0 and 1"):
        calibrator.transform([-0.1, 0.5])
    assert np.isfinite(calibrator.transform([0, 1])).all()


@pytest.mark.skipif(importlib.util.find_spec("xgboost") is None, reason="XGBoost not installed")
def test_xgboost_real_fit_and_native_attributions():
    data = example_data()
    model = fit_model("xgboost", *data, config={"param_grid": [
        {"max_depth": 2, "n_estimators": 8, "learning_rate": 0.2}
    ]})
    probabilities = model.predict_proba(data[3], data[5])
    contributions = model.contributions(data[3], data[5])
    np.testing.assert_allclose(expit(contributions.sum(axis=1)), probabilities, atol=1e-6)
    assert np.std(probabilities) > 0
    assert model.params["selected_params"]["tree_method"] == "hist"


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch not installed")
def test_gru_is_causal_padding_invariant_and_roundtrips(tmp_path):
    data = example_data()
    model = fit_model("gru", *data, config={"hidden_size_grid": [4], "epochs": 2,
                                           "patience": 2, "batch_size": 8})
    # A shorter patient's probabilities must not depend on later rows, another
    # patient's longer sequence, or padding used to put them in the same batch.
    short = data[3].iloc[:2].reset_index(drop=True)
    long = data[3].iloc[4:12].reset_index(drop=True)
    alone = model.predict_proba(short, ["patient_a"] * 2)
    together = model.predict_proba(pd.concat([short, long], ignore_index=True),
                                   ["patient_a"] * 2 + ["patient_b"] * 8)
    full_sequence = model.predict_proba(data[3].iloc[:4], ["patient_a"] * 4)
    np.testing.assert_allclose(alone, together[:2], atol=1e-7)
    np.testing.assert_allclose(alone, full_sequence[:2], atol=1e-7)
    assert model.contributions(short, ["patient_a"] * 2) is None
    artifact = save_model(model, tmp_path / "gru.joblib")
    loaded = load_model(artifact)
    np.testing.assert_allclose(loaded.predict_proba(short, ["patient_a"] * 2), alone, atol=1e-7)
    assert model.params["best_epoch"] >= 1
    assert model.params["bidirectional"] is False
