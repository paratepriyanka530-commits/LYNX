"""XGBoost candidate model with the common Pipeline 2 model interface."""

from .xgboost_model import LYNXXGBoost


def get_model_info():
    return {
        "name": "xgboost",
        "display_name": "XGBoost",
        "supported_tasks": ["classification", "regression"],
    }


def build_model(task_type, random_state=42):
    if task_type not in {"classification", "regression"}:
        raise ValueError(f"Unsupported task type for XGBoost: {task_type}")
    model = LYNXXGBoost(task_type=task_type)
    model.model.set_params(random_state=random_state)
    return model


def train_model(model, X_train, y_train):
    return model.train(X_train, y_train)


def predict_model(model, X_test):
    return model.predict(X_test)
