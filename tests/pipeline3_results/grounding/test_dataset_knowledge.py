from __future__ import annotations

import pandas as pd

from pipelines.pipeline3_results.grounding.compatibility import adapt_grounding_result
from pipelines.pipeline3_results.grounding.dataset_knowledge import build_dataset_knowledge
from pipelines.pipeline3_results.grounding.engine import ground_semantic_intent
from pipelines.pipeline3_results.grounding.similarity import (
    LexicalSemanticSimilarityProvider,
    SemanticSimilarityProvider,
)
from pipelines.pipeline3_results.nlp.compatibility_adapter import adapt_semantic_intent
from pipelines.pipeline3_results.nlp.semantic_intent import SemanticIntent


def _report(*, confidence: float = 0.9) -> dict:
    return {
        "semantic_profile": {
            "dataset_context": {
                "row_entity": "transaction",
                "row_entity_confidence": 0.88,
                "dataset_description": "One row represents a transaction.",
            },
            "columns": [
                {
                    "column_name": "amount",
                    "semantic_type": "currency",
                    "semantic_role": "measurement",
                    "value_function": "measurement",
                    "numeric_meaning": "magnitude",
                    "concept_description": "Revenue amount for the transaction",
                    "reasoning_summary": "A monetary measurement.",
                    "confidence": confidence,
                    "samples": ["RAW_VALUE_MUST_NOT_REACH_PROVIDER"],
                },
                {
                    "column_name": "event_date",
                    "semantic_type": "date",
                    "semantic_role": "timestamp",
                    "value_function": "temporal",
                    "concept_description": "Date of the transaction",
                    "confidence": 0.8,
                },
            ],
        },
        "before_profile": {
            "column_profiles": [
                {
                    "column_name": "amount",
                    "physical_dtype": "float64",
                    "missing": {"count": 2},
                    "uniqueness": {"ratio": 0.7},
                },
                {"column_name": "event_date", "physical_dtype": "datetime64[ns]"},
            ]
        },
        "quality_summary": {"quality_score": 0.94},
    }


def _intent(concept: str = "revenue", *, goal: str = "SIMPLE_AGGREGATION") -> SemanticIntent:
    return SemanticIntent(
        intent="INSIGHT",
        analytical_goal=goal,
        measure_concepts=[concept],
        aggregation="SUM" if goal == "SIMPLE_AGGREGATION" else "NONE",
    )


class RecordingSimilarityProvider(SemanticSimilarityProvider):
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.fallback = LexicalSemanticSimilarityProvider()

    def similarity(self, query_text: str, candidate_text: str) -> float:
        self.calls.append((query_text, candidate_text))
        return self.fallback.similarity(query_text, candidate_text)


def test_dataset_knowledge_preserves_provenance_and_quality_without_raw_values() -> None:
    knowledge = build_dataset_knowledge(_report(confidence=0.0), dataset_id="dataset-1")

    assert knowledge.dataset_id == "dataset-1"
    assert knowledge.row_entity == "transaction"
    assert knowledge.get_column("AMOUNT").confidence == 0.0
    assert knowledge.get_column("amount").quality_metadata["missing"] == {"count": 2}
    assert knowledge.quality == {"quality_score": 0.94}
    assert knowledge.metadata_availability == {
        "pipeline1_report": True,
        "dataset_context": True,
        "column_semantics": True,
        "column_profiles": True,
        "quality": True,
        "schema_fallback": False,
    }
    assert "RAW_VALUE_MUST_NOT_REACH_PROVIDER" not in knowledge.model_dump_json()


def test_missing_metadata_is_distinct_from_low_confidence_metadata() -> None:
    missing = build_dataset_knowledge(None, dataframe_schema={"amount": "float64"})
    low_confidence = build_dataset_knowledge(_report(confidence=0.0))

    assert missing.metadata_availability["column_semantics"] is False
    assert missing.get_column("amount").confidence is None
    assert low_confidence.metadata_availability["column_semantics"] is True
    assert low_confidence.get_column("amount").confidence == 0.0


def test_similarity_provider_receives_metadata_text_only() -> None:
    provider = RecordingSimilarityProvider()
    result = ground_semantic_intent(
        _intent(), build_dataset_knowledge(_report()), similarity_provider=provider
    )

    assert result.status == "RESOLVED"
    assert provider.calls
    assert all(isinstance(value, str) for call in provider.calls for value in call)
    assert all("RAW_VALUE_MUST_NOT_REACH_PROVIDER" not in value for call in provider.calls for value in call)


def test_production_adapter_prefers_dataset_knowledge_when_semantics_exist() -> None:
    dataframe = pd.DataFrame({"amount": [1.0], "event_date": pd.to_datetime(["2026-01-01"])})
    result = adapt_semantic_intent(
        _intent(), "total revenue", dataframe, pipeline1_report=_report()
    )

    assert result.status == "RESOLVED"
    assert result.query.resolved_measures == ["amount"]
    assert "grounding:dataset_knowledge" in result.signals
    assert "grounding:source:LEGACY_FALLBACK" not in result.signals


def test_legacy_fallback_is_explicit_when_semantic_metadata_is_missing() -> None:
    dataframe = pd.DataFrame({"revenue": [1.0, 2.0]})
    result = adapt_semantic_intent(_intent(), "total revenue", dataframe)

    assert result.status == "RESOLVED"
    assert "grounding:source:LEGACY_FALLBACK" in result.signals
    assert "grounding:legacy_fallback_reason:semantic_metadata_unavailable" in result.signals


def test_grounded_statistic_stops_at_controlled_compatibility_boundary() -> None:
    semantic = _intent("revenue standard deviation", goal="DISTRIBUTION")
    grounding = ground_semantic_intent(semantic, build_dataset_knowledge(_report()))
    result = adapt_grounding_result(semantic, grounding, "revenue standard deviation")

    assert grounding.status == "RESOLVED"
    assert grounding.measures[0].resolved_expression.kind == "STATISTIC"
    assert result.status == "NEEDS_CLARIFICATION"
    assert "GROUNDING_CORRECT" in result.signals
    assert "EXECUTION_UNSUPPORTED_CURRENTLY" in result.signals
