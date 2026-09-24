"""LightGBM candidate model with the common Pipeline 2 model interface."""

from .lightgbm_model import LYNXLightGBM
from .cb_chl import CBCHLConfigurationError


def get_model_info():
    return {
        "name": "lightgbm",
        "display_name": "LightGBM",
        "supported_tasks": ["classification", "regression"],
    }


def build_model(
    task_type,
    random_state=42,
    training_mode="standard",
    cb_chl_eligible=False,
    cost_config=None,
):
    if task_type not in {"classification", "regression"}:
        raise ValueError(f"Unsupported task type for LightGBM: {task_type}")
    if task_type == "regression" and training_mode != "standard":
        raise CBCHLConfigurationError("CB-CHL is classification-only.")
    if training_mode == "cb_chl" and not cb_chl_eligible:
        raise CBCHLConfigurationError(
            "CB-CHL cannot activate because this dataset is not marked "
            "cb_chl_eligible."
        )
    if training_mode == "cb_chl" and cost_config is None:
        raise CBCHLConfigurationError(
            "CB-CHL requires an explicit cost configuration; standard "
            "LightGBM remains available with training_mode='standard'."
        )
    model = LYNXLightGBM(
        training_mode=training_mode,
        cb_chl_eligible=cb_chl_eligible,
        cost_config=cost_config,
        task_type=task_type,
    )
    model.model.set_params(random_state=random_state)
    return model


def train_model(model, X_train, y_train):
    return model.train(X_train, y_train)


def predict_model(model, X_test):
    return model.predict(X_test)
