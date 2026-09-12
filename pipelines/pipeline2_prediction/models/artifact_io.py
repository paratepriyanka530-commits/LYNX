"""Compatibility loading for artifacts serialized before the repository merge."""

from contextlib import contextmanager
from datetime import datetime, timezone
from importlib import import_module
import pickle
from pathlib import Path
import sys

import joblib


_MISSING = object()
REFERENCE_ARTIFACT_ROOT = Path(__file__).resolve().parents[1] / "reference_artifacts"


def reference_artifact_dir(model_id):
    """Return the model-specific integration/reference artifact directory."""

    return REFERENCE_ARTIFACT_ROOT / model_id


def reference_artifact_path(model_id, filename):
    """Return a path under the uniform reference-artifact tree."""

    return reference_artifact_dir(model_id) / filename


def training_artifact_dir(model_id, dataset_id, run_id=None, artifact_root=None):
    """Return an isolated directory for one dataset/model training run."""

    def safe_component(value):
        cleaned = "".join(
            character if character.isalnum() or character in {"-", "_"} else "_"
            for character in str(value)
        ).strip("_")
        if not cleaned:
            raise ValueError("Artifact path components cannot be empty")
        return cleaned

    effective_run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path(artifact_root) if artifact_root is not None else REFERENCE_ARTIFACT_ROOT / "training_runs"
    return root / safe_component(dataset_id) / safe_component(effective_run_id) / safe_component(model_id)


@contextmanager
def module_aliases(alias_targets):
    """Temporarily map explicit serialized module names to current modules."""

    replacements = {}
    for legacy_name, current_name in alias_targets.items():
        replacements[legacy_name] = sys.modules.get(legacy_name, _MISSING)
        sys.modules[legacy_name] = import_module(current_name)

    try:
        yield
    finally:
        for legacy_name, previous in replacements.items():
            if previous is _MISSING:
                sys.modules.pop(legacy_name, None)
            else:
                sys.modules[legacy_name] = previous


@contextmanager
def _legacy_module_aliases(module_names):
    """Temporarily map staged model module paths to this package."""

    alias_targets = {}
    for legacy_name in module_names:
        suffix = legacy_name.removeprefix("models")
        current_name = f"pipelines.pipeline2_prediction.models{suffix}"
        alias_targets[legacy_name] = current_name

    with module_aliases(alias_targets):
        yield


def load_joblib_artifact(path, legacy_modules=()):
    """Load a joblib artifact, remapping only explicitly listed legacy paths."""

    module_names = ["models", *legacy_modules]
    with _legacy_module_aliases(module_names):
        return joblib.load(path)


def load_pickle_artifact(path):
    """Load a pickle, remapping CUDA tensor storage when CUDA is unavailable."""

    import torch

    if torch.cuda.is_available():
        with open(path, "rb") as artifact_file:
            return pickle.load(artifact_file)

    original_torch_load = torch.load

    def cpu_torch_load(*args, **kwargs):
        kwargs.setdefault("map_location", torch.device("cpu"))
        return original_torch_load(*args, **kwargs)

    torch.load = cpu_torch_load
    try:
        with open(path, "rb") as artifact_file:
            return pickle.load(artifact_file)
    finally:
        torch.load = original_torch_load
