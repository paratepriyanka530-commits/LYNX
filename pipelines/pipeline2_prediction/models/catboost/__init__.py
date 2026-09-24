"""CatBoost candidate model with the common Pipeline 2 model interface."""

from .catboost_model import LYNXCatBoost


def get_model_info():
    return {
        "name": "catboost",
        "display_name": "CatBoost",
        "supported_tasks": ["classification", "regression"],
    }


def build_model(task_type, random_state=42):
    if task_type not in {"classification", "regression"}:
        raise ValueError(f"Unsupported task type for CatBoost: {task_type}")
    model = LYNXCatBoost(task_type=task_type)
    model.model.set_params(random_seed=random_state)
    return model


def train_model(model, X_train, y_train):
    return model.train(X_train, y_train)


def predict_model(model, X_test):
    return model.predict(X_test)
