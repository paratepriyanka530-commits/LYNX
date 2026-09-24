"""Optional CRUMB execution strategy around the unchanged TabICLv2 wrapper."""

from ..crumb import CrumbConfig, CrumbModel
from .. import tabiclv2 as base


def get_model_info():
    return {
        "name": "tabiclv2_crumb",
        "display_name": "TabICLv2 + CRUMB",
        "base_model_id": "tabiclv2",
        "execution_strategy": "crumb",
        "supported_tasks": ["classification", "regression"],
        "experimental": True,
    }


def build_model(task_type, random_state=42, **crumb_overrides):
    # The paper's TabICLv2 experiments cap the context at its 4,096-sample
    # window. This is variant-local and does not add a base-model hard gate.
    settings = {"random_state": random_state, "max_context_size": 4_096}
    settings.update(crumb_overrides)
    config = CrumbConfig(**settings)
    return CrumbModel(
        base.build_model(task_type, random_state=random_state),
        base.train_model,
        base.predict_model,
        config,
    )


def train_model(model, X_train, y_train):
    return model.fit(X_train, y_train)


def predict_model(model, X_test):
    return model.predict(X_test)


__all__ = ["get_model_info", "build_model", "train_model", "predict_model"]
