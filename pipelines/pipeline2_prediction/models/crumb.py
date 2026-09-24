"""CRUMB context selection for optional prior-fitted-model execution.

This implements Algorithm 1 and the RFF/batched-greedy acceleration from
Heredge et al. (2026).  It is deliberately a strategy wrapper: base-model
packages and their standard execution paths are not modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class CrumbConfig:
    n_clusters: int = 20
    context_fraction: float = 0.10
    max_context_size: int = 10_000
    rff_dimension: int = 64
    greedy_batch_size: int = 50
    random_state: int = 42
    bandwidth_sample_size: int = 2_048
    early_stopping: bool = True
    early_stop_interval: int = 50
    early_stop_epsilon: float = 1e-4
    early_stop_patience: int = 5

    def __post_init__(self) -> None:
        if self.n_clusters < 1 or not 0 < self.context_fraction <= 1:
            raise ValueError("CRUMB requires n_clusters >= 1 and 0 < context_fraction <= 1")
        if min(self.max_context_size, self.rff_dimension, self.greedy_batch_size) < 1:
            raise ValueError("CRUMB context, RFF, and greedy-batch sizes must be positive")


def _numeric_matrix(X: Any, name: str) -> np.ndarray:
    """Return finite numeric features; do not invent mixed-type distances."""
    if isinstance(X, pd.DataFrame):
        non_numeric = [column for column in X if not pd.api.types.is_numeric_dtype(X[column])]
        if non_numeric:
            raise TypeError(
                f"CRUMB selector requires numeric features; {name} has non-numeric "
                f"columns {non_numeric}. The paper does not specify a mixed-type encoding."
            )
        values = X.to_numpy(dtype=np.float64, copy=True)
    else:
        values = np.asarray(X, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite before CRUMB context selection")
    return values


def _median_bandwidth(train: np.ndarray, test: np.ndarray, config: CrumbConfig) -> float:
    combined = np.vstack([train, test])
    # The paper specifies the median heuristic but not its large-N computation.
    # A fixed-seed, bounded sample keeps this implementation auditable and avoids
    # an otherwise quadratic auxiliary allocation.
    if len(combined) > config.bandwidth_sample_size:
        rng = np.random.default_rng(config.random_state)
        chosen = np.sort(
            rng.choice(len(combined), config.bandwidth_sample_size, replace=False)
        )
        combined = combined[chosen]
    distances = pairwise_distances(combined, metric="euclidean")
    positive = distances[np.triu_indices_from(distances, k=1)]
    positive = positive[positive > 0]
    return float(np.median(positive)) if len(positive) else 1.0


def _rff(train: np.ndarray, test: np.ndarray, sigma: float, config: CrumbConfig):
    rng = np.random.default_rng(config.random_state)
    omega = rng.normal(
        0.0, 1.0 / sigma, size=(train.shape[1], config.rff_dimension)
    )
    phase = rng.uniform(0.0, 2.0 * np.pi, size=config.rff_dimension)
    scale = np.sqrt(2.0 / config.rff_dimension)
    return scale * np.cos(train @ omega + phase), scale * np.cos(test @ omega + phase)


def greedy_mmd_indices(
    train_embedding: np.ndarray,
    cluster_embedding: np.ndarray,
    budget: int,
    config: CrumbConfig,
) -> np.ndarray:
    """Select a CRUMB context with the paper's batched kernel-herding score."""
    budget = min(int(budget), len(train_embedding))
    cluster_mean = cluster_embedding.mean(axis=0)
    affinity = train_embedding @ cluster_mean
    selected: list[int] = []
    selected_mask = np.zeros(len(train_embedding), dtype=bool)
    selected_sum = np.zeros(train_embedding.shape[1], dtype=np.float64)
    previous_mmd: float | None = None
    stale_checks = 0
    warmup = max(100, int(np.ceil(budget / 10)))

    while len(selected) < budget:
        t = len(selected)
        redundancy = train_embedding @ selected_sum
        scores = -2.0 * affinity + (2.0 / (t + 1)) * redundancy
        scores[selected_mask] = np.inf
        take = min(config.greedy_batch_size, budget - t)
        candidates = np.flatnonzero(~selected_mask)
        ordered = sorted(candidates, key=lambda index: (scores[index], int(index)))
        batch = np.asarray(ordered[:take], dtype=int)
        selected.extend(batch.tolist())
        selected_mask[batch] = True
        selected_sum += train_embedding[batch].sum(axis=0)

        current = len(selected)
        if (
            config.early_stopping
            and current >= warmup
            and current % config.early_stop_interval == 0
        ):
            selected_mean = selected_sum / current
            mmd_squared = float(np.sum((cluster_mean - selected_mean) ** 2))
            if previous_mmd is not None:
                relative = (previous_mmd - mmd_squared) / max(previous_mmd, np.finfo(float).eps)
                stale_checks = stale_checks + 1 if relative < config.early_stop_epsilon else 0
                if stale_checks >= config.early_stop_patience:
                    break
            previous_mmd = mmd_squared
    return np.asarray(selected, dtype=int)


class CrumbModel:
    """Store train context, then select/furnish one context per test cluster."""

    def __init__(
        self,
        base_model: Any,
        train_base: Callable[[Any, Any, Any], Any],
        predict_base: Callable[[Any, Any], Any],
        config: CrumbConfig,
    ) -> None:
        self.base_model = base_model
        self.train_base = train_base
        self.predict_base = predict_base
        self.config = config
        self.X_train_: Any | None = None
        self.y_train_: Any | None = None
        self.last_selection_: list[dict[str, Any]] = []
        self.scaler_: StandardScaler | None = None

    def fit(self, X_train: Any, y_train: Any) -> "CrumbModel":
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train lengths differ")
        _numeric_matrix(X_train, "X_train")
        self.X_train_ = X_train.copy() if hasattr(X_train, "copy") else np.array(X_train, copy=True)
        self.y_train_ = y_train.copy() if hasattr(y_train, "copy") else np.array(y_train, copy=True)
        return self

    @staticmethod
    def _take_rows(values: Any, indices: np.ndarray) -> Any:
        return values.iloc[indices] if hasattr(values, "iloc") else np.asarray(values)[indices]

    def predict(self, X_test: Any) -> np.ndarray:
        if self.X_train_ is None or self.y_train_ is None:
            raise RuntimeError("CRUMB must receive training context before prediction")
        train_values = _numeric_matrix(self.X_train_, "X_train")
        test_values = _numeric_matrix(X_test, "X_test")
        if train_values.shape[1] != test_values.shape[1]:
            raise ValueError("CRUMB train/test feature counts differ")
        if len(test_values) == 0:
            return np.asarray([])

        # Leakage invariant: the scaler is fitted exclusively on X_train.
        self.scaler_ = StandardScaler().fit(train_values)
        train_scaled = self.scaler_.transform(train_values)
        test_scaled = self.scaler_.transform(test_values)
        cluster_count = min(self.config.n_clusters, len(test_values))
        labels = KMeans(
            n_clusters=cluster_count,
            init="k-means++",
            algorithm="lloyd",
            n_init="auto",
            random_state=self.config.random_state,
        ).fit_predict(test_scaled)
        sigma = _median_bandwidth(train_scaled, test_scaled, self.config)
        train_rff, test_rff = _rff(train_scaled, test_scaled, sigma, self.config)
        budget = min(
            len(train_values),
            self.config.max_context_size,
            max(1, int(np.ceil(self.config.context_fraction * len(train_values)))),
        )

        output = np.empty(len(test_values), dtype=object)
        self.last_selection_ = []
        for cluster_id in range(cluster_count):
            test_indices = np.flatnonzero(labels == cluster_id)
            context_indices = greedy_mmd_indices(
                train_rff, test_rff[test_indices], budget, self.config
            )
            X_context = self._take_rows(self.X_train_, context_indices)
            y_context = self._take_rows(self.y_train_, context_indices)
            X_cluster = self._take_rows(X_test, test_indices)
            self.train_base(self.base_model, X_context, y_context)
            predictions = np.asarray(self.predict_base(self.base_model, X_cluster)).reshape(-1)
            if len(predictions) != len(test_indices):
                raise ValueError("Base model returned the wrong CRUMB cluster prediction shape")
            output[test_indices] = predictions
            self.last_selection_.append({
                "cluster_id": cluster_id,
                "test_indices": test_indices.tolist(),
                "context_indices": context_indices.tolist(),
                "context_size": len(context_indices),
            })
        try:
            return output.astype(np.asarray(predictions).dtype, copy=False)
        except (TypeError, ValueError):
            return output


__all__ = ["CrumbConfig", "CrumbModel", "greedy_mmd_indices"]
