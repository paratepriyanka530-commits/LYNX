"""LightGBM candidate model with the common Pipeline 2 model interface."""

from .lightgbm_model import LYNXLightGBM


def get_model_info():
    return {
        "name": "lightgbm",
        "display_name": "LightGBM",
        "supported_tasks": ["classification"],
    }


def build_model(task_type, random_state=42):
    if task_type != "classification":
        raise ValueError(f"Unsupported task type for LightGBM: {task_type}")
    model = LYNXLightGBM()
    model.model.set_params(random_state=random_state)
    return model


def train_model(model, X_train, y_train):
    return model.train(X_train, y_train)


def predict_model(model, X_test):
    return model.predict(X_test)
