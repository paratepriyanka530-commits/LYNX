"""Optional CRUMB execution strategy around the unchanged TabPFN wrapper."""

from ..crumb import CrumbConfig, CrumbModel
from .. import tabpfn as base


def get_model_info():
    return {
        "name": "tabpfn_crumb",
        "display_name": "TabPFN + CRUMB",
        "base_model_id": "tabpfn",
        "execution_strategy": "crumb",
        "supported_tasks": ["classification"],
        "experimental": True,
    }


def build_model(task_type, random_state=42, device=None, **crumb_overrides):
    settings = {"random_state": random_state, "max_context_size": 10_000}
    settings.update(crumb_overrides)
    config = CrumbConfig(**settings)
    return CrumbModel(
        base.build_model(task_type, random_state=random_state, device=device),
        base.train_model,
        base.predict_model,
        config,
    )


def train_model(model, X_train, y_train):
    return model.fit(X_train, y_train)


def predict_model(model, X_test):
    return model.predict(X_test)


__all__ = ["get_model_info", "build_model", "train_model", "predict_model"]
