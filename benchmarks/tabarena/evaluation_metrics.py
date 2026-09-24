"""Deterministic predictive and calibration metrics for Pipeline 2 runs."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


ECE_N_BINS = 15
ECE_METHOD = "top_label_equal_width"
PROBABILITY_TOLERANCE = 1e-6


PROBABILITY_CAPABILITIES = {
    "random_forest": {
        "supports_classification_probabilities": True,
        "probability_source": "sklearn Pipeline.predict_proba",
        "probability_class_order_available": True,
    },
    "xgboost": {
        "supports_classification_probabilities": True,
        "probability_source": "LYNXXGBoost.predict_probability",
        "probability_class_order_available": True,
    },
    "lightgbm": {
        "supports_classification_probabilities": True,
        "probability_source": "LYNXLightGBM.predict_probability",
        "probability_class_order_available": True,
    },
    "catboost": {
        "supports_classification_probabilities": True,
        "probability_source": "LYNXCatBoost.predict_probability",
        "probability_class_order_available": True,
    },
    "xtab": {
        "supports_classification_probabilities": True,
        "probability_source": "XTabCandidate.predict_proba",
        "probability_class_order_available": True,
    },
    "tabpfn": {
        "supports_classification_probabilities": True,
        "probability_source": "LYNXTabPFN.predict_probability",
        "probability_class_order_available": True,
    },
    "tabm": {
        "supports_classification_probabilities": True,
        "probability_source": "PytabkitCandidate.predict_proba (TabM-D)",
        "probability_class_order_available": True,
    },
    "tabiclv2": {
        "supports_classification_probabilities": True,
        "probability_source": "PytabkitCandidate.predict_proba (TabICLv2)",
        "probability_class_order_available": True,
    },
    "realmlp": {
        "supports_classification_probabilities": True,
        "probability_source": "PytabkitCandidate.predict_proba (RealMLP-TD)",
        "probability_class_order_available": True,
    },
    "ft_transformer": {
        "supports_classification_probabilities": True,
        "probability_source": "PytabkitCandidate.predict_proba (FT-Transformer)",
        "probability_class_order_available": True,
    },
}


def _class_order(model: Any, model_id: str) -> np.ndarray | None:
    candidates = [
        getattr(model, "classes_", None),
        getattr(getattr(model, "model", None), "classes_", None),
        getattr(getattr(model, "estimator", None), "classes_", None),
    ]
    for value in candidates:
        if value is not None:
            return np.asarray(value).reshape(-1)
    if model_id == "xtab":
        predictor = getattr(model, "predictor", None)
        mapping = getattr(getattr(predictor, "preprocessor", None), "target_mapping", None)
        if mapping:
            return np.asarray([
                label for label, _ in sorted(mapping.items(), key=lambda item: item[1])
            ])
    return None


def _align_class_order(reported: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """Resolve representation-only differences such as XTab's string labels."""
    if len(reported) != len(expected):
        return reported
    aligned = []
    remaining = list(expected)
    for value in reported:
        exact = [candidate for candidate in remaining if candidate == value]
        matches = exact or [candidate for candidate in remaining if str(candidate) == str(value)]
        if len(matches) != 1:
            return reported
        aligned.append(matches[0])
        remaining.remove(matches[0])
    return np.asarray(aligned)


def obtain_probabilities(
    model_id: str,
    model: Any,
    X_test: Any,
    expected_classes: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, str, str | None]:
    capability = PROBABILITY_CAPABILITIES.get(model_id)
    if capability is None or not capability["supports_classification_probabilities"]:
        return None, None, "unavailable", "No reliable probability path is registered."
    try:
        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(X_test), dtype=float)
        elif hasattr(model, "predict_probability"):
            probabilities = np.asarray(model.predict_probability(X_test), dtype=float)
        else:
            return None, None, "unavailable", "Registered model exposes no fitted probability method."
        classes = _class_order(model, model_id)
        if classes is None:
            return probabilities, None, "invalid", "Fitted probability class ordering is unavailable."
        classes = _align_class_order(classes, np.asarray(expected_classes))
        if probabilities.ndim == 1 and len(classes) == 2:
            probabilities = np.column_stack([1.0 - probabilities, probabilities])
        return probabilities, classes, "available", None
    except Exception as exc:
        return None, None, "unavailable", f"{type(exc).__name__}: {exc}"


def validate_probabilities(
    probabilities: Any,
    class_order: Any,
    y_true: Any,
    expected_classes: Any,
    tolerance: float = PROBABILITY_TOLERANCE,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    try:
        values = np.asarray(probabilities, dtype=float)
    except (TypeError, ValueError) as exc:
        return None, None, f"Probabilities are not numeric: {exc}"
    classes = None if class_order is None else np.asarray(class_order).reshape(-1)
    y = np.asarray(y_true).reshape(-1)
    expected = np.asarray(expected_classes).reshape(-1)
    if values.ndim != 2:
        return None, None, f"Probability array must be 2D, got shape {values.shape}."
    if values.shape[0] != len(y):
        return None, None, f"Probability rows={values.shape[0]} do not match y_test rows={len(y)}."
    if classes is None:
        return None, None, "Probability class ordering is unavailable."
    if values.shape[1] != len(classes):
        return None, None, "Probability column count does not match class ordering."
    if len(classes) != len(expected) or set(classes.tolist()) != set(expected.tolist()):
        return None, None, "Probability class order does not match the expected fitted classes."
    if len(set(classes.tolist())) != len(classes):
        return None, None, "Probability class ordering contains duplicates."
    if not set(y.tolist()).issubset(set(classes.tolist())):
        return None, None, "y_test contains labels absent from probability class ordering."
    if not np.isfinite(values).all():
        return None, None, "Probability array contains NaN or infinite values."
    if (values < -tolerance).any() or (values > 1.0 + tolerance).any():
        return None, None, "Probability values fall outside [0,1] tolerance."
    if not np.allclose(values.sum(axis=1), 1.0, atol=tolerance, rtol=0.0):
        return None, None, "Probability rows do not sum to 1 within tolerance."
    return np.clip(values, 0.0, 1.0), classes, None


def brier_score(y_true: Any, probabilities: np.ndarray, classes: np.ndarray) -> float:
    y = np.asarray(y_true).reshape(-1)
    if len(classes) == 2:
        return float(np.mean((probabilities[:, 1] - (y == classes[1]).astype(float)) ** 2))
    indices = {label: index for index, label in enumerate(classes.tolist())}
    one_hot = np.zeros_like(probabilities, dtype=float)
    one_hot[np.arange(len(y)), [indices[label] for label in y.tolist()]] = 1.0
    # Deliberately not divided by the number of classes.
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    y_true: Any,
    probabilities: np.ndarray,
    classes: np.ndarray,
    n_bins: int = ECE_N_BINS,
) -> float:
    if n_bins != ECE_N_BINS:
        raise ValueError(f"Pipeline 2 freezes ECE at exactly {ECE_N_BINS} bins")
    y = np.asarray(y_true).reshape(-1)
    confidence = probabilities.max(axis=1)
    predicted = classes[np.argmax(probabilities, axis=1)]
    correct = (predicted == y).astype(float)
    # [0,1/15), ... [14/15,1]; confidence=1 belongs to the final bin.
    bin_index = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for index in range(n_bins):
        members = bin_index == index
        if members.any():
            ece += float(members.mean()) * abs(
                float(correct[members].mean()) - float(confidence[members].mean())
            )
    return float(ece)


def classification_results(
    y_train: Any,
    y_test: Any,
    predictions: Any,
    model_id: str,
    model: Any,
    X_test: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    y_train_array = np.asarray(y_train).reshape(-1)
    y = np.asarray(y_test).reshape(-1)
    predicted = np.asarray(predictions).reshape(-1)
    expected_classes = np.unique(y_train_array)
    predictive = {
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "precision_macro": float(precision_score(y, predicted, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, predicted, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "f2_macro": float(fbeta_score(y, predicted, beta=2, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y, predicted, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y, predicted, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y, predicted, average="weighted", zero_division=0)),
        "roc_auc": None,
        "log_loss": None,
        "mcc": float(matthews_corrcoef(y, predicted)),
        "confusion_matrix": confusion_matrix(y, predicted, labels=expected_classes).tolist(),
        "metric_reasons": {},
    }
    probabilities, classes, status, reason = obtain_probabilities(
        model_id, model, X_test, expected_classes
    )
    probability_metadata = {
        "status": status,
        "reason": reason,
        "source": (PROBABILITY_CAPABILITIES.get(model_id) or {}).get("probability_source"),
        "class_order": classes.tolist() if classes is not None else None,
    }
    if status == "available":
        probabilities, classes, invalid_reason = validate_probabilities(
            probabilities, classes, y, expected_classes
        )
        if invalid_reason:
            status, reason = "invalid", invalid_reason
            probability_metadata.update({"status": status, "reason": reason})
    if status != "available":
        predictive["metric_reasons"].update({"roc_auc": reason, "log_loss": reason})
        calibration = {
            "brier_score": None, "ece": None, "ece_n_bins": ECE_N_BINS,
            "ece_method": ECE_METHOD, "status": status, "reason": reason,
        }
        return predictive, calibration, probability_metadata

    try:
        if len(classes) == 2:
            predictive["roc_auc"] = float(roc_auc_score((y == classes[1]).astype(int), probabilities[:, 1]))
        else:
            predictive["roc_auc"] = float(roc_auc_score(
                y, probabilities, labels=classes, multi_class="ovr", average="macro"
            ))
    except ValueError as exc:
        predictive["metric_reasons"]["roc_auc"] = f"ValueError: {exc}"
    try:
        predictive["log_loss"] = float(log_loss(y, probabilities, labels=classes))
    except ValueError as exc:
        predictive["metric_reasons"]["log_loss"] = f"ValueError: {exc}"
    calibration = {
        "brier_score": brier_score(y, probabilities, classes),
        "ece": expected_calibration_error(y, probabilities, classes),
        "ece_n_bins": ECE_N_BINS,
        "ece_method": ECE_METHOD,
        "status": "available",
        "reason": None,
    }
    return predictive, calibration, probability_metadata


def regression_results(y_test: Any, predictions: Any):
    y = np.asarray(y_test, dtype=float).reshape(-1)
    predicted = np.asarray(predictions, dtype=float).reshape(-1)
    rmse = float(mean_squared_error(y, predicted) ** 0.5)
    target_std = float(np.std(y))
    predictive = {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y, predicted)),
        "r2": float(r2_score(y, predicted)),
        "nrmse": float(rmse / target_std) if target_std > 0 else None,
        "nrmse_normalizer": "population standard deviation of y_test",
        "metric_reasons": {} if target_std > 0 else {"nrmse": "y_test has zero variance"},
    }
    calibration = {
        "brier_score": None,
        "ece": None,
        "ece_n_bins": None,
        "ece_method": None,
        "status": "not_applicable",
        "reason": "point-prediction regression protocol",
    }
    probability_metadata = {
        "status": "not_applicable", "reason": "regression task",
        "source": None, "class_order": None,
    }
    return predictive, calibration, probability_metadata


__all__ = [
    "ECE_METHOD", "ECE_N_BINS", "PROBABILITY_CAPABILITIES",
    "brier_score", "classification_results", "expected_calibration_error",
    "obtain_probabilities", "regression_results", "validate_probabilities",
]
