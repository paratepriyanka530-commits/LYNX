"""TabM candidate using pytabkit's maintained sklearn interface."""

from pytabkit import TabM_D_Classifier, TabM_D_Regressor

from ..pytabkit_candidate import PytabkitCandidate


def get_model_info():
    return {
        "name": "tabm",
        "display_name": "TabM-D",
        "supported_tasks": ["classification", "regression"],
        "gpu_requirement": "optional; CPU and MPS are also supported",
        "preprocessing": (
            "train-fitted numeric imputation in LYNX; pytabkit internally fits "
            "categorical one-hot encoding and quantile-style numeric transforms"
        ),
    }


def build_model(task_type, random_state=42, **config_overrides):
    classes = {
        "classification": TabM_D_Classifier,
        "regression": TabM_D_Regressor,
    }
    if task_type not in classes:
        raise ValueError(f"Unsupported task type for TabM: {task_type}")
    estimator = classes[task_type](random_state=random_state, **config_overrides)
    return PytabkitCandidate(estimator)


def train_model(model, X_train, y_train):
    return model.fit(X_train, y_train)


def predict_model(model, X_test):
    return model.predict(X_test)


__all__ = ["get_model_info", "build_model", "train_model", "predict_model"]
