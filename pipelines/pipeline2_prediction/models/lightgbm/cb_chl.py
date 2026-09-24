"""Paper-faithful binary CB-CHL objective for optional LightGBM experiments.

The implementation follows Zhao, Liu, and Zhao (IEEE Access 2024),
Equations (10), (12), (16), (18), and (23), plus Algorithm 1.  It deliberately
contains no default cost matrix: false-positive and false-negative costs are a
domain/research input, not something that can be inferred from class imbalance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class CBCHLConfigurationError(ValueError):
    """Raised when the experimental CB-CHL mode is not safely configured."""


@dataclass(frozen=True)
class CBCHLTrainingState:
    """Auditable quantities derived only from the labels passed to ``train``."""

    n_samples: int
    positive_count: int
    negative_count: int
    negative_class_balance_factor: float


@dataclass(frozen=True)
class CBCHLObjective:
    """Pickle-safe LightGBM callable bound to one training split."""

    labels: np.ndarray
    costs: np.ndarray

    def __call__(self, y_true: np.ndarray, raw_scores: np.ndarray):
        observed = np.asarray(y_true, dtype=float).reshape(-1)
        if not np.array_equal(observed, self.labels):
            raise RuntimeError(
                "CB-CHL objective received labels/order different from its "
                "train-only configuration."
            )
        return cb_chl_grad_hess(raw_scores, observed, self.costs)


def validate_cost_matrix(cost_config: Any, n_samples: int) -> np.ndarray:
    """Return an ``(n_samples, 2)`` [C_fp, C_fn] matrix or fail explicitly."""

    if cost_config is None:
        raise CBCHLConfigurationError(
            "CB-CHL requires an explicit per-training-instance cost matrix "
            "with columns [false_positive_cost, false_negative_cost]. No "
            "anneal/MIC cost matrix is inferred or supplied by default."
        )
    if isinstance(cost_config, dict):
        expected = {"false_positive_costs", "false_negative_costs"}
        if set(cost_config) != expected:
            raise CBCHLConfigurationError(
                "CB-CHL cost configuration must contain exactly "
                "false_positive_costs and false_negative_costs."
            )
        matrix = np.column_stack(
            (cost_config["false_positive_costs"], cost_config["false_negative_costs"])
        )
    else:
        matrix = np.asarray(cost_config)
    try:
        matrix = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError) as exc:
        raise CBCHLConfigurationError("CB-CHL costs must be numeric.") from exc
    if matrix.shape != (n_samples, 2):
        raise CBCHLConfigurationError(
            f"CB-CHL cost matrix must have shape ({n_samples}, 2); got {matrix.shape}."
        )
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise CBCHLConfigurationError(
            "CB-CHL costs must be finite and non-negative."
        )
    if np.any(np.all(matrix == 0, axis=1)):
        raise CBCHLConfigurationError(
            "Each CB-CHL training row must have at least one non-zero error cost."
        )
    return matrix


def training_state(y_true: Any) -> tuple[np.ndarray, CBCHLTrainingState]:
    """Validate binary labels and derive the paper's S1/S0 factor."""

    y = np.asarray(y_true, dtype=float).reshape(-1)
    if not np.isin(y, (0.0, 1.0)).all():
        raise CBCHLConfigurationError(
            "The published CB-CHL objective is binary-only and requires labels 0/1."
        )
    positive_count = int(np.sum(y == 1.0))
    negative_count = int(np.sum(y == 0.0))
    if positive_count == 0 or negative_count == 0:
        raise CBCHLConfigurationError(
            "CB-CHL training labels must contain both binary classes."
        )
    state = CBCHLTrainingState(
        n_samples=len(y),
        positive_count=positive_count,
        negative_count=negative_count,
        negative_class_balance_factor=positive_count / negative_count,
    )
    return y, state


def cb_chl_loss(
    raw_scores: Any,
    y_true: Any,
    cost_matrix: Any,
    *,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Return per-instance Equation (16) loss values."""

    y, state = training_state(y_true)
    costs = validate_cost_matrix(cost_matrix, len(y))
    z = np.asarray(raw_scores, dtype=float).reshape(-1)
    if z.shape != y.shape:
        raise ValueError(f"raw_scores must have shape {y.shape}; got {z.shape}.")
    probability = 1.0 / (1.0 + np.exp(-np.clip(z, -709.0, 709.0)))
    signed_y = 2.0 * y - 1.0
    p_star = probability * (signed_y + 1.0) / 2.0 + (
        1.0 - probability
    ) * (1.0 - signed_y) / 2.0
    p_star = np.clip(p_star, epsilon, 1.0 - epsilon)
    alpha = y + state.negative_class_balance_factor * (1.0 - y)
    zeta = costs[:, 1] * y + costs[:, 0] * (1.0 - y)
    return -alpha * zeta * (1.0 - p_star) ** 2 * np.log(p_star)


def cb_chl_grad_hess(
    raw_scores: Any,
    y_true: Any,
    cost_matrix: Any,
    *,
    epsilon: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the paper's Algorithm 1 gradient and Hessian."""

    y, state = training_state(y_true)
    costs = validate_cost_matrix(cost_matrix, len(y))
    z = np.asarray(raw_scores, dtype=float).reshape(-1)
    if z.shape != y.shape:
        raise ValueError(f"raw_scores must have shape {y.shape}; got {z.shape}.")
    probability = 1.0 / (1.0 + np.exp(-np.clip(z, -709.0, 709.0)))
    signed_y = 2.0 * y - 1.0
    p_star = probability * (signed_y + 1.0) / 2.0 + (
        1.0 - probability
    ) * (1.0 - signed_y) / 2.0
    p_star = np.clip(p_star, epsilon, 1.0 - epsilon)
    alpha = y + state.negative_class_balance_factor * (1.0 - y)
    zeta = costs[:, 1] * y + costs[:, 0] * (1.0 - y)

    one_minus = 1.0 - p_star
    v = 2.0 * p_star * np.log(p_star) + p_star - 1.0
    u = alpha * zeta * signed_y * one_minus**2
    grad = u * v
    du = -2.0 * alpha * zeta * signed_y * one_minus
    dv = 2.0 * np.log(p_star) + 3.0
    hess = (v * du + u * dv) * signed_y * p_star * one_minus
    return grad, hess


def make_cb_chl_objective(y_train: Any, cost_config: Any):
    """Bind train-only class counts and per-row costs to a LightGBM objective."""

    y, state = training_state(y_train)
    costs = validate_cost_matrix(cost_config, len(y)).copy()

    return CBCHLObjective(labels=y.copy(), costs=costs), state
