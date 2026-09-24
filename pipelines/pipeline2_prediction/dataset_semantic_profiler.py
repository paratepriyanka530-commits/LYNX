"""One-call dataset-level semantic evidence and schema for Pipeline 2."""

from __future__ import annotations

import json
from typing import Annotated, Any

from pydantic import BaseModel, Field, StringConstraints


MAX_REPRESENTATIVE_FEATURE_NAMES = 80
NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Pipeline2DatasetSemanticProfile(BaseModel):
    """Validated holistic semantics used by the model selector."""

    probable_domain: NonEmptyText
    subdomain: NonEmptyText
    dataset_purpose: NonEmptyText
    row_entity: NonEmptyText
    target_concept: NonEmptyText
    feature_context: list[NonEmptyText] = Field(min_length=1, max_length=12)
    semantic_summary: NonEmptyText
    domain_confidence: float = Field(ge=0.0, le=1.0)


def representative_feature_names(
    feature_names: list[str], cap: int = MAX_REPRESENTATIVE_FEATURE_NAMES,
) -> list[str]:
    """Select a deterministic, order-preserving sample spread over all columns."""
    if cap < 2:
        raise ValueError("Representative feature-name cap must be at least 2")
    if len(feature_names) <= cap:
        return list(feature_names)
    last = len(feature_names) - 1
    return [feature_names[index * last // (cap - 1)] for index in range(cap)]


def build_dataset_semantic_evidence(
    *, dataset_name: str, task_type: str, target_column: str,
    row_count: int, feature_count: int, feature_type_counts: dict[str, int],
    feature_names: list[str],
) -> dict[str, Any]:
    if target_column in feature_names:
        raise ValueError("Target column must not appear in predictor feature names")
    if feature_count != len(feature_names):
        raise ValueError("Feature count does not match the supplied predictor names")
    return {
        "dataset_name": dataset_name,
        "task_type": task_type,
        "target_column": target_column,
        "row_count": int(row_count),
        "feature_count": int(feature_count),
        "feature_type_counts": {
            key: int(feature_type_counts[key]) for key in (
                "numeric", "categorical", "boolean", "datetime", "other"
            )
        },
        "representative_feature_names": representative_feature_names(feature_names),
        "representative_feature_name_count": min(
            feature_count, MAX_REPRESENTATIVE_FEATURE_NAMES
        ),
        "representative_sampling": (
            "all_predictors_in_source_order" if feature_count <= MAX_REPRESENTATIVE_FEATURE_NAMES
            else "80_order_preserving_evenly_spread_predictors"
        ),
    }


def build_dataset_semantic_prompt(evidence: dict[str, Any]) -> str:
    serialized = json.dumps(evidence, ensure_ascii=False, sort_keys=True)
    return f"""
You are the dataset-level semantic profiler for the LYNX Pipeline 2 model
selector. Infer holistic dataset context from the compact observed metadata
below. This is one dataset-level assessment, not per-column semantic typing.

Return the required structured fields:
- probable_domain: broad domain such as healthcare, finance, biology, or unknown;
- subdomain: a more specific area, or unknown when unsupported;
- dataset_purpose: what prediction/analysis task the table appears to support;
- row_entity: what one row represents;
- target_concept: what the target variable represents;
- feature_context: 1-12 short broad feature groups, never a column-by-column list;
- semantic_summary: concise synthesis of domain, row entity, predictors, and target;
- domain_confidence: calibrated number in [0,1].

Treat dataset and column names as evidence, not proof. Do not invent provenance,
measurement units, populations, collection procedures, or causal claims. When
the evidence is opaque, say "unknown" and lower domain_confidence. Numeric or
encoded feature names alone do not establish a domain. The representative names
are deterministically spread across the predictor columns; the benchmark target
is excluded from that list.

COMPACT DATASET EVIDENCE
{serialized}

Return only the structured object required by the supplied JSON schema.
""".strip()
