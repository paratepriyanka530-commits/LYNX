"""XTab candidate model adapted to Pipeline 2's common model interface."""

import pandas as pd

from ..artifact_io import reference_artifact_path
from .xtab_config import XTabConfig
from .xtab_predictor import XTabPredictor
from .xtab_trainer import XTabTrainer


DEFAULT_ARTIFACT_PATH = reference_artifact_path("xtab", "xtab.pt")
DEFAULT_PRETRAINED_BACKBONE_PATH = reference_artifact_path(
    "xtab", "official_iter_2k_backbone.ckpt"
)


def get_model_info():
    return {
        "name": "xtab",
        "display_name": "XTab",
        "supported_tasks": ["classification", "regression"],
        "minimum_training_rows": 10,
        "hard_maximum_rows": None,
        "hard_maximum_features": None,
        "hard_maximum_classes": None,
        "pretraining": (
            "official XTab iter_2k cross-table FT-Transformer backbone; "
            "all 46 backbone tensors loaded before downstream optimization"
        ),
        "preprocessing": "fitted preprocessor persisted inside the torch checkpoint",
    }


class XTabCandidate:
    """Estimator-style adapter around the incoming trainer and predictor."""

    def __init__(self, task_type, random_state=42, **config_overrides):
        self.task_type = task_type
        self.random_state = random_state
        self.config_overrides = config_overrides
        self.trainer = None
        self.predictor = None

    def fit(self, X_train, y_train):
        frame = X_train.copy() if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train)
        target_column = "__lynx_xtab_target__"
        while target_column in frame.columns:
            target_column += "_"
        frame[target_column] = pd.Series(y_train).reset_index(drop=True).to_numpy()
        self.config_overrides.setdefault(
            "pretrained_backbone_path", str(DEFAULT_PRETRAINED_BACKBONE_PATH)
        )
        config = XTabConfig(
            task=self.task_type,
            random_seed=self.random_state,
            **self.config_overrides,
        )
        self.trainer = XTabTrainer(config)
        model = self.trainer.train(frame, target_column)
        self.predictor = XTabPredictor(
            model=model,
            preprocessor=self.trainer.preprocessor,
            device=self.trainer.device,
        )
        return self

    def predict(self, X_test):
        if self.predictor is None:
            raise RuntimeError("XTabCandidate must be fitted before prediction")
        frame = X_test if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test)
        return self.predictor.predict(frame)

    def predict_proba(self, X_test):
        if self.predictor is None:
            raise RuntimeError("XTabCandidate must be fitted before prediction")
        frame = X_test if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test)
        return self.predictor.predict_proba(frame)


def build_model(task_type, random_state=42, **config_overrides):
    if task_type not in {"classification", "regression"}:
        raise ValueError(f"Unsupported task type for XTab: {task_type}")
    return XTabCandidate(task_type, random_state, **config_overrides)


def train_model(model, X_train, y_train):
    return model.fit(X_train, y_train)


def predict_model(model, X_test):
    return model.predict(X_test)


def load_trained_model(path=DEFAULT_ARTIFACT_PATH, device=None):
    return XTabPredictor.load(path, device=device)
