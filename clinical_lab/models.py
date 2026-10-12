"""Train-only preprocessing, patient-separated model tuning, and calibration.

Rows within each patient must arrive in chronological order. Temporal models are
unidirectional and reset their recurrent state for each patient. All outputs are
research probabilities; calibration must be fitted on a separate patient split.

Joblib files are executable pickle artifacts: load only trusted local files.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler


_FAMILY_ALIASES = {
    "lr": "logistic", "logistic_regression": "logistic",
    "logistic": "logistic", "xgb": "xgboost", "xgboost": "xgboost",
    "gru": "gru",
}


def _labels(values: Any, length: int, *, name: str, both_classes: bool = True) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or len(result) != length:
        raise ValueError(f"{name} must be one-dimensional and match the feature rows")
    if not np.isfinite(result).all() or not np.isin(result, [0, 1]).all():
        raise ValueError(f"{name} must contain finite binary labels (0 or 1)")
    if both_classes and len(np.unique(result)) != 2:
        raise ValueError(f"{name} requires both positive and negative labels")
    return result


def _patient_ids(values: Any, length: int) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or len(result) != length:
        raise ValueError("patient_ids must be one-dimensional and match the feature rows")
    if pd.isna(result).any():
        raise ValueError("patient_ids cannot be missing")
    return result.astype(str)


def _numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.shape[1] == 0:
        raise ValueError("X must be a DataFrame with at least one feature")
    if frame.columns.duplicated().any():
        raise ValueError("Feature columns must be unique")
    if not all(isinstance(column, str) for column in frame.columns):
        raise ValueError("Feature column names must be strings")
    try:
        values = frame.to_numpy(dtype=np.float64, na_value=np.nan)
    except (ValueError, TypeError) as exc:
        raise ValueError("All model features must be numeric") from exc
    if np.isinf(values).any():
        raise ValueError("Feature values cannot contain infinity")
    return pd.DataFrame(values, columns=frame.columns, index=frame.index)


class FeaturePreprocessor:
    """Median imputation and optional scaling, learned on training rows only."""

    def __init__(self, *, scale: bool):
        self.scale = scale
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scaler = StandardScaler() if scale else None
        self.feature_columns: list[str] = []

    def fit(self, X: pd.DataFrame) -> "FeaturePreprocessor":
        frame = _numeric_frame(X)
        if frame.empty:
            raise ValueError("Training features cannot be empty")
        self.feature_columns = list(frame.columns)
        transformed = self.imputer.fit_transform(frame)
        if self.scaler is not None:
            self.scaler.fit(transformed)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        frame = _numeric_frame(X)
        if set(frame.columns) != set(self.feature_columns):
            raise ValueError("Prediction feature columns differ from the training schema")
        if len(frame) == 0:
            return np.empty((0, len(self.feature_columns)), dtype=np.float32)
        transformed = self.imputer.transform(frame.loc[:, self.feature_columns])
        if self.scaler is not None:
            transformed = self.scaler.transform(transformed)
        return np.asarray(transformed, dtype=np.float32)


class TabularModel:
    """Shared one-dimensional probability API for sklearn and XGBoost models."""

    def __init__(self, family: str, estimator: Any, preprocessor: FeaturePreprocessor,
                 params: dict[str, Any]):
        self.family = family
        self.estimator = estimator
        self.preprocessor = preprocessor
        self.feature_columns = list(preprocessor.feature_columns)
        self.params = params

    def predict_proba(self, X: pd.DataFrame, patient_ids: Any = None) -> np.ndarray:
        if patient_ids is not None:
            _patient_ids(patient_ids, len(X))
        values = self.preprocessor.transform(X)
        if len(values) == 0:
            return np.empty(0, dtype=float)
        return self.estimator.predict_proba(values)[:, 1].astype(float)

    def contributions(self, X: pd.DataFrame, patient_ids: Any = None) -> pd.DataFrame:
        """Additive contributions in uncalibrated log-odds, including ``bias``.

        These explain a model score, not causal effects or clinical reasons.
        Logistic contributions are standardized feature * coefficient. XGBoost
        uses its native TreeSHAP calculation. Neither explains calibration.
        """
        if patient_ids is not None:
            _patient_ids(patient_ids, len(X))
        values = self.preprocessor.transform(X)
        columns = self.feature_columns + ["bias"]
        if "bias" in self.feature_columns:
            raise ValueError("The reserved attribution name 'bias' is a feature column")
        if len(values) == 0:
            return pd.DataFrame(columns=columns, index=X.index, dtype=float)
        if self.family == "logistic":
            contributions = values * self.estimator.coef_[0]
            bias = np.full((len(values), 1), float(self.estimator.intercept_[0]))
            contributions = np.column_stack([contributions, bias])
        else:
            from xgboost import DMatrix
            contributions = self.estimator.get_booster().predict(
                DMatrix(values), pred_contribs=True
            )
        result = pd.DataFrame(contributions, columns=columns, index=X.index)
        result.attrs["scale"] = "uncalibrated_log_odds"
        result.attrs["method"] = "coefficient_times_feature" if self.family == "logistic" else "TreeSHAP"
        return result


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required to train or load a GRU model") from exc
    return torch


def _make_network(input_size: int, hidden_size: int):
    torch = _torch()

    class CausalGRUNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = torch.nn.GRU(input_size, hidden_size, batch_first=True,
                                    bidirectional=False)
            self.output = torch.nn.Linear(hidden_size, 1)

        def forward(self, features, lengths):
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                features, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.gru(packed)
            output, _ = torch.nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True, total_length=features.shape[1]
            )
            return self.output(output).squeeze(-1)

    return CausalGRUNetwork()


def _groups(patient_ids: np.ndarray) -> list[np.ndarray]:
    """Group rows while preserving the input order within each trajectory."""
    groups: dict[str, list[int]] = {}
    for row, patient in enumerate(patient_ids):
        groups.setdefault(str(patient), []).append(row)
    return [np.asarray(indices, dtype=np.int64) for indices in groups.values()]


def _sequence_batch(values: np.ndarray, groups: list[np.ndarray], labels=None):
    torch = _torch()
    lengths = torch.tensor([len(indices) for indices in groups], dtype=torch.long)
    width = int(lengths.max())
    features = torch.zeros((len(groups), width, values.shape[1]), dtype=torch.float32)
    targets = torch.zeros((len(groups), width), dtype=torch.float32)
    mask = torch.zeros((len(groups), width), dtype=torch.bool)
    for row, indices in enumerate(groups):
        features[row, :len(indices)] = torch.from_numpy(values[indices])
        mask[row, :len(indices)] = True
        if labels is not None:
            targets[row, :len(indices)] = torch.as_tensor(labels[indices], dtype=torch.float32)
    return features, lengths, targets, mask


class GRUModel:
    """CPU, causal GRU over variable-length patient trajectories."""

    family = "gru"

    def __init__(self, network: Any, preprocessor: FeaturePreprocessor,
                 hidden_size: int, batch_size: int, params: dict[str, Any]):
        self.network = network
        self.preprocessor = preprocessor
        self.feature_columns = list(preprocessor.feature_columns)
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.params = params

    def _predict_values(self, values: np.ndarray, patient_ids: np.ndarray) -> np.ndarray:
        torch = _torch()
        predictions = np.empty(len(values), dtype=float)
        groups = _groups(patient_ids)
        self.network.eval()
        with torch.no_grad():
            for start in range(0, len(groups), self.batch_size):
                batch_groups = groups[start:start + self.batch_size]
                features, lengths, _, _ = _sequence_batch(values, batch_groups)
                batch_scores = torch.sigmoid(self.network(features, lengths)).cpu().numpy()
                for row, indices in enumerate(batch_groups):
                    predictions[indices] = batch_scores[row, :len(indices)]
        return predictions

    def predict_proba(self, X: pd.DataFrame, patient_ids: Any) -> np.ndarray:
        ids = _patient_ids(patient_ids, len(X))
        return self._predict_values(self.preprocessor.transform(X), ids)

    def contributions(self, X: pd.DataFrame, patient_ids: Any = None) -> None:
        """Unavailable: the GRU does not expose a validated feature attribution."""
        return None

    def __getstate__(self):
        state = self.__dict__.copy()
        network = state.pop("network")
        state["network_state"] = {
            name: tensor.detach().cpu().numpy().copy()
            for name, tensor in network.state_dict().items()
        }
        return state

    def __setstate__(self, state):
        torch = _torch()
        network_state = state.pop("network_state")
        self.__dict__.update(state)
        self.network = _make_network(len(self.feature_columns), self.hidden_size)
        self.network.load_state_dict({name: torch.from_numpy(value) for name, value in network_state.items()})
        self.network.eval()


class SigmoidCalibrator:
    """Logistic calibration of clipped model log-odds on held-out patients."""

    def __init__(self, estimator: LogisticRegression):
        self.estimator = estimator
        self.method = "sigmoid_on_logit"

    @staticmethod
    def _logits(probabilities: Any) -> np.ndarray:
        values = np.asarray(probabilities, dtype=float)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("Probabilities must be a finite one-dimensional array")
        if ((values < 0) | (values > 1)).any():
            raise ValueError("Probabilities must lie between 0 and 1")
        clipped = np.clip(values, 1e-7, 1 - 1e-7)
        return (np.log(clipped) - np.log1p(-clipped)).reshape(-1, 1)

    def transform(self, probabilities: Any) -> np.ndarray:
        logits = self._logits(probabilities)
        if len(logits) == 0:
            return np.empty(0, dtype=float)
        return self.estimator.predict_proba(logits)[:, 1]

    predict = transform
    predict_proba = transform


def fit_calibrator(probabilities: Any, labels: Any) -> SigmoidCalibrator:
    """Fit only on a dedicated calibration split, never tune/test predictions."""
    logits = SigmoidCalibrator._logits(probabilities)
    targets = _labels(labels, len(logits), name="Calibration labels")
    estimator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000, random_state=2019)
    estimator.fit(logits, targets)
    return SigmoidCalibrator(estimator)


def _config(config: dict[str, Any] | None, family: str) -> dict[str, Any]:
    config = config or {}
    if family in config and isinstance(config[family], dict):
        return dict(config[family])
    if "models" in config and isinstance(config["models"], dict):
        return dict(config["models"].get(family, {}))
    return dict(config)


def _fit_gru(train_values, train_labels, train_ids, tune_values, tune_labels,
             tune_ids, preprocessor, cfg, seed):
    torch = _torch()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    sizes = cfg.get("hidden_size_grid", [cfg["hidden_size"]] if "hidden_size" in cfg else [16, 32])
    epochs = int(cfg.get("epochs", 20))
    patience = int(cfg.get("patience", 4))
    batch_size = int(cfg.get("batch_size", 32))
    learning_rate = float(cfg.get("learning_rate", 0.001))
    if not sizes or epochs < 1 or patience < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("GRU grid and training settings must be positive")
    groups = _groups(train_ids)
    positive_weight = (len(train_labels) - train_labels.sum()) / train_labels.sum()
    positive_weight = float(cfg.get("pos_weight", positive_weight))
    best_model = None
    best_score = -np.inf
    tuning_scores = []
    for hidden_size in sizes:
        hidden_size = int(hidden_size)
        if hidden_size < 1:
            raise ValueError("GRU hidden sizes must be positive")
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        network = _make_network(train_values.shape[1], hidden_size)
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate,
                                     weight_decay=float(cfg.get("weight_decay", 1e-4)))
        loss_function = torch.nn.BCEWithLogitsLoss(
            reduction="none", pos_weight=torch.tensor(positive_weight, dtype=torch.float32)
        )
        params = {"hidden_size": hidden_size, "batch_size": batch_size,
                  "learning_rate": learning_rate, "max_epochs": epochs,
                  "patience": patience, "pos_weight": positive_weight,
                  "bidirectional": False, "seed": seed}
        model = GRUModel(network, preprocessor, hidden_size, batch_size, params)
        best_epoch_score = -np.inf
        best_state = None
        best_epoch = 0
        epochs_without_improvement = 0
        history = []
        for epoch in range(epochs):
            network.train()
            total_loss = 0.0
            total_rows = 0
            for start in range(0, len(groups), batch_size):
                # The permutation is constructed once per epoch below and never
                # changes chronological order within an individual patient.
                if start == 0:
                    order = rng.permutation(len(groups))
                batch_groups = [groups[index] for index in order[start:start + batch_size]]
                features, lengths, labels, mask = _sequence_batch(train_values, batch_groups, train_labels)
                optimizer.zero_grad(set_to_none=True)
                logits = network(features, lengths)
                row_losses = loss_function(logits, labels)
                loss = row_losses[mask].mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=5.0)
                optimizer.step()
                total_loss += float(loss.detach()) * int(mask.sum())
                total_rows += int(mask.sum())
            score = float(average_precision_score(tune_labels, model._predict_values(tune_values, tune_ids)))
            history.append({"epoch": epoch + 1, "train_loss": total_loss / total_rows,
                            "tune_average_precision": score})
            if score > best_epoch_score + 1e-8:
                best_epoch_score = score
                best_epoch = epoch + 1
                best_state = deepcopy(network.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break
        network.load_state_dict(best_state)
        network.eval()
        params["best_epoch"] = best_epoch
        params["history"] = history
        tuning_scores.append({"hidden_size": hidden_size, "best_epoch": best_epoch,
                              "average_precision": best_epoch_score})
        if best_epoch_score > best_score:
            best_score = best_epoch_score
            best_model = model
    best_model.params["tuning_scores"] = tuning_scores
    best_model.params["selection_metric"] = "tune_average_precision"
    best_model.params["tune_average_precision"] = best_score
    return best_model


def fit_model(family: str, X_train: pd.DataFrame, y_train: Any, ids_train: Any,
              X_tune: pd.DataFrame, y_tune: Any, ids_tune: Any,
              config: dict[str, Any] | None = None, seed: int = 2019):
    """Fit a family and select hyperparameters by tuning average precision only.

    Training and tuning patient IDs must be disjoint. This function does not
    inspect calibration or test data, refit on tuning data, or fit thresholds.
    ``config`` can be direct, keyed by family, or under a ``models`` mapping.
    """
    try:
        family = _FAMILY_ALIASES[family.lower()]
    except (KeyError, AttributeError) as exc:
        raise ValueError("Model family must be logistic, xgboost, or gru") from exc
    train_labels = _labels(y_train, len(X_train), name="Training labels")
    tune_labels = _labels(y_tune, len(X_tune), name="Tuning labels")
    train_ids = _patient_ids(ids_train, len(X_train))
    tune_ids = _patient_ids(ids_tune, len(X_tune))
    if set(train_ids) & set(tune_ids):
        raise ValueError("Training and tuning patients must be disjoint")
    cfg = _config(config, family)
    preprocessor = FeaturePreprocessor(scale=family != "xgboost").fit(X_train)
    train_values = preprocessor.transform(X_train)
    tune_values = preprocessor.transform(X_tune)
    if family == "gru":
        return _fit_gru(train_values, train_labels, train_ids, tune_values,
                        tune_labels, tune_ids, preprocessor, cfg, seed)
    if family == "logistic":
        strengths = cfg.get("C_grid", [cfg["C"]] if "C" in cfg else [0.1, 1.0, 10.0])
        candidates = [{"C": float(value)} for value in strengths]
        if any(candidate["C"] <= 0 for candidate in candidates):
            raise ValueError("Logistic C values must be positive")
    else:
        candidates = cfg.get("param_grid", [
            {"max_depth": 3, "n_estimators": 120, "learning_rate": 0.05},
            {"max_depth": 4, "n_estimators": 180, "learning_rate": 0.05},
        ])
        if not isinstance(candidates, list) or not all(isinstance(item, dict) for item in candidates):
            raise ValueError("XGBoost param_grid must be a list of parameter dictionaries")
    if not candidates:
        raise ValueError("The hyperparameter grid must not be empty")
    best_estimator, best_params, best_score = None, None, -np.inf
    tuning_scores = []
    for candidate in candidates:
        if family == "logistic":
            params = {"C": candidate["C"], "max_iter": int(cfg.get("max_iter", 1000)),
                      "class_weight": cfg.get("class_weight", "balanced"),
                      "solver": "lbfgs", "random_state": seed}
            estimator = LogisticRegression(**params)
        else:
            try:
                from xgboost import XGBClassifier
            except ImportError as exc:
                raise ImportError("XGBoost is required to train an xgboost model") from exc
            params = {"max_depth": 3, "n_estimators": 120, "learning_rate": 0.05,
                      "subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 1.0}
            params.update(candidate)
            for name in ("n_estimators", "max_depth", "learning_rate", "subsample",
                         "colsample_bytree", "reg_lambda", "min_child_weight", "scale_pos_weight"):
                if name in cfg:
                    params[name] = cfg[name]
            # Preserve the reproducible CPU classification task even if a grid
            # accidentally includes device/objective/thread parameters.
            params.update({"objective": "binary:logistic", "eval_metric": "logloss",
                           "tree_method": "hist", "device": "cpu",
                           "n_jobs": 2, "random_state": seed})
            estimator = XGBClassifier(**params)
        estimator.fit(train_values, train_labels)
        score = float(average_precision_score(tune_labels, estimator.predict_proba(tune_values)[:, 1]))
        tuning_scores.append({"params": dict(params), "average_precision": score})
        if score > best_score:
            best_estimator, best_params, best_score = estimator, dict(params), score
    metadata = {"selected_params": best_params, "tuning_scores": tuning_scores,
                "selection_metric": "tune_average_precision", "tune_average_precision": best_score,
                "seed": seed}
    return TabularModel(family, best_estimator, preprocessor, metadata)


def save_model(model: Any, path: str | Path) -> Path:
    """Serialize a fitted local model/calibrator. Never commit patient data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)
    return path


def load_model(path: str | Path) -> Any:
    """Load a TRUSTED local joblib artifact; untrusted pickle may execute code."""
    return joblib.load(Path(path))
