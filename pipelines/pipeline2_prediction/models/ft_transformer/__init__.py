"""Non-pretrained FT-Transformer candidate from pytabkit."""

from pytabkit import FTT_D_Classifier, FTT_D_Regressor

from ..pytabkit_candidate import PytabkitCandidate


def get_model_info():
    return {
        "name": "ft_transformer",
        "display_name": "FT-Transformer (non-pretrained control)",
        "supported_tasks": ["classification", "regression"],
        "minimum_training_rows": 2,
        "hard_maximum_rows": None,
        "hard_maximum_features": None,
        "hard_maximum_classes": None,
        "gpu_requirement": "optional; CPU and MPS are supported",
        "pretraining": "none; fresh random initialization for every dataset",
        "purpose": (
            "non-pretrained FT-Transformer baseline; intended architectural "
            "control for an XTab pretraining ablation"
        ),
        "preprocessing": (
            "LYNX train-fitted null handling/schema followed by pytabkit's "
            "internally fitted FT-Transformer feature transforms"
        ),
    }


def build_model(task_type, random_state=42, **config_overrides):
    classes = {"classification": FTT_D_Classifier, "regression": FTT_D_Regressor}
    if task_type not in classes:
        raise ValueError(f"Unsupported task type for FT-Transformer: {task_type}")
    return PytabkitCandidate(classes[task_type](random_state=random_state, **config_overrides))


def train_model(model, X_train, y_train):
    return model.fit(X_train, y_train)


def predict_model(model, X_test):
    return model.predict(X_test)


__all__ = ["get_model_info", "build_model", "train_model", "predict_model"]
