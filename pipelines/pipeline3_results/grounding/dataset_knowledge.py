"""Normalize Pipeline 1 evidence into one DatasetKnowledge contract."""

from __future__ import annotations

from typing import Any, Mapping

from pipelines.pipeline3_results.grounding.schemas import (
    DatasetColumnKnowledge,
    DatasetKnowledge,
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _column_entries(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    semantic_profile = _mapping(report.get("semantic_profile"))
    entries = semantic_profile.get("columns", [])
    if not entries and "columns" in report:
        entries = report.get("columns", [])
    return [item for item in entries if isinstance(item, Mapping)]


def _profile_entries(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    profile = _mapping(report.get("before_profile"))
    entries = profile.get("column_profiles", [])
    return {
        str(item.get("column_name")): item
        for item in entries
        if isinstance(item, Mapping) and item.get("column_name") is not None
    }


def build_dataset_knowledge(
    pipeline1_report: Mapping[str, Any] | None,
    *,
    dataset_id: str | None = None,
    dataframe_schema: Mapping[str, str] | None = None,
) -> DatasetKnowledge:
    """Build normalized metadata without reading or copying dataframe rows."""
    report = _mapping(pipeline1_report)
    semantic_profile = _mapping(report.get("semantic_profile"))
    dataset_context = _mapping(semantic_profile.get("dataset_context"))
    profiles = _profile_entries(report)
    semantics = {
        str(item.get("column_name")): item
        for item in _column_entries(report)
        if item.get("column_name") is not None
    }

    ordered_names: list[str] = []
    for source in (dataframe_schema or {}, profiles, semantics):
        for name in source:
            if name not in ordered_names:
                ordered_names.append(name)

    columns: list[DatasetColumnKnowledge] = []
    for name in ordered_names:
        semantic = _mapping(semantics.get(name))
        profile = _mapping(profiles.get(name))
        quality_metadata = {
            key: profile[key]
            for key in (
                "row_count", "non_null_count", "missing", "uniqueness",
                "repetition", "numeric_statistics",
            )
            if key in profile
        }
        physical_dtype = profile.get("physical_dtype")
        if physical_dtype is None and dataframe_schema is not None:
            physical_dtype = dataframe_schema.get(name)
        columns.append(DatasetColumnKnowledge(
            name=name,
            physical_dtype=str(physical_dtype) if physical_dtype is not None else None,
            semantic_type=(str(semantic["semantic_type"]) if semantic.get("semantic_type") is not None else None),
            semantic_role=(str(semantic["semantic_role"]) if semantic.get("semantic_role") is not None else None),
            value_function=(str(semantic["value_function"]) if semantic.get("value_function") is not None else None),
            numeric_meaning=(str(semantic["numeric_meaning"]) if semantic.get("numeric_meaning") is not None else None),
            concept_description=(str(semantic["concept_description"]) if semantic.get("concept_description") else None),
            reasoning_summary=(str(semantic["reasoning_summary"]) if semantic.get("reasoning_summary") else None),
            confidence=(float(semantic["confidence"]) if semantic.get("confidence") is not None else None),
            referenced_entity=(str(semantic["referenced_entity"]) if semantic.get("referenced_entity") else None),
            entity_alignment=(str(semantic["entity_alignment"]) if semantic.get("entity_alignment") else None),
            quality_metadata=quality_metadata,
        ))

    return DatasetKnowledge(
        dataset_id=dataset_id,
        dataset_description=(
            str(dataset_context["dataset_description"])
            if dataset_context.get("dataset_description") else None
        ),
        row_entity=(str(dataset_context["row_entity"]) if dataset_context.get("row_entity") else None),
        row_entity_confidence=(
            float(dataset_context["row_entity_confidence"])
            if dataset_context.get("row_entity_confidence") is not None else None
        ),
        columns=columns,
        quality=dict(_mapping(report.get("quality_summary"))),
        metadata_availability={
            "pipeline1_report": bool(report),
            "dataset_context": bool(dataset_context),
            "column_semantics": bool(semantics),
            "column_profiles": bool(profiles),
            "quality": bool(_mapping(report.get("quality_summary"))),
            "schema_fallback": bool(dataframe_schema),
        },
    )


__all__ = ["build_dataset_knowledge"]
