"""TabPFN candidate model with the common Pipeline 2 model interface."""

from .tabpfn_model import LYNXTabPFN, TabPFNProvisioningError


def get_model_info():
    return {
        "name": "tabpfn",
        "display_name": "TabPFN V3",
        "supported_tasks": ["classification", "regression"],
    }


def build_model(task_type, random_state=42, device=None):
    if task_type not in {"classification", "regression"}:
        raise ValueError(f"Unsupported task type for TabPFN: {task_type}")
    model = LYNXTabPFN(model_path="auto", device=device, task_type=task_type)
    model.model.set_params(random_state=random_state)
    return model


def train_model(model, X_train, y_train):
    return model.train(X_train, y_train)


def predict_model(model, X_test):
    return model.predict(X_test)
