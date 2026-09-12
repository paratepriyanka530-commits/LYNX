"""TabICLv2 candidate using the official ``tabicl`` sklearn interface."""

from tabicl import TabICLClassifier, TabICLRegressor

from ..pytabkit_candidate import PytabkitCandidate


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
    }


def build_model(task_type, random_state=42, **config_overrides):
    classes = {"classification": TabICLClassifier, "regression": TabICLRegressor}
    if task_type not in classes:
        raise ValueError(f"Unsupported task type for TabICLv2: {task_type}")
    return PytabkitCandidate(classes[task_type](random_state=random_state, **config_overrides))


def train_model(model, X_train, y_train):
    return model.fit(X_train, y_train)


def predict_model(model, X_test):
    return model.predict(X_test)


__all__ = ["get_model_info", "build_model", "train_model", "predict_model"]
