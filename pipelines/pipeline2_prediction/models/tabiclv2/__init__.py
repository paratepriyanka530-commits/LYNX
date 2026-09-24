"""TabICLv2 candidate using the official ``tabicl`` sklearn interface."""

from tabicl import TabICLClassifier, TabICLRegressor

from ..pytabkit_candidate import PytabkitCandidate


class TabICLProvisioningError(RuntimeError):
    """The official TabICLv2 checkpoint is unavailable."""


def get_model_info():
    return {
        "name": "tabiclv2",
        "display_name": "TabICLv2",
        "supported_tasks": ["classification", "regression"],
        "minimum_training_rows": 2,
        "hard_maximum_rows": None,
        "hard_maximum_features": None,
        "hard_maximum_classes": None,
        "gpu_requirement": "optional; GPU recommended for larger tables",
        "preprocessing": (
            "LYNX train-fitted null handling/schema followed by TabICLv2's "
            "internally fitted categorical encoding, scaling, and normalization"
        ),
        "checkpoint": (
            "official tabicl v2 classifier/regressor checkpoint, downloaded and "
            "cached by the upstream package"
        ),
        "authentication_requirement": "none; checkpoints are public on Hugging Face",
    }


def build_model(task_type, random_state=42, **config_overrides):
    classes = {"classification": TabICLClassifier, "regression": TabICLRegressor}
    if task_type not in classes:
        raise ValueError(f"Unsupported task type for TabICLv2: {task_type}")
    return PytabkitCandidate(classes[task_type](random_state=random_state, **config_overrides))


def train_model(model, X_train, y_train):
    try:
        return model.fit(X_train, y_train)
    except OSError as exc:
        raise TabICLProvisioningError(
            "TabICLv2 requires its official classification or regression "
            "checkpoint. Provide a populated Hugging Face cache or permit the "
            "public jingang/TabICL download; no authentication is required."
        ) from exc


def predict_model(model, X_test):
    return model.predict(X_test)


__all__ = [
    "TabICLProvisioningError",
    "get_model_info",
    "build_model",
    "train_model",
    "predict_model",
]
