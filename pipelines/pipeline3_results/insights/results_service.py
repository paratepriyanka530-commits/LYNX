"""Application-independent composition service for deterministic Insights."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from pipelines.pipeline3_results.insights.analysis_engine import (
    execute_insight_query,
)
from pipelines.pipeline3_results.insights.query_adapter import adapt_insight_query
from pipelines.pipeline3_results.insights.query_validator import (
    validate_insight_query,
)
from pipelines.pipeline3_results.insights.response_builder import (
    build_controlled_response,
    build_insight_response,
)
from pipelines.pipeline3_results.nlp.orchestrator import interpret_query
from pipelines.pipeline3_results.nlp.llm.semantic_interpreter import (
    interpret_query_with_fallback,
)
from intelligence.llm.provider import LLMProvider
from pipelines.pipeline3_results.visualization.column_classifier import (
    classify_dataframe,
)
from pipelines.pipeline3_results.visualization.schemas import DatasetClassification


def build_insight_results(
    dataframe: pd.DataFrame,
    question: str,
    pipeline1_report: Mapping[str, Any] | None = None,
    dataset_name: str | None = None,
    classification: DatasetClassification | None = None,
    llm_provider: LLMProvider | None = None,
) -> dict[str, Any]:
    """Interpret, validate, compute, and phrase one deterministic data answer."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    if pipeline1_report is not None and not isinstance(pipeline1_report, Mapping):
        raise TypeError("pipeline1_report must be a Mapping or None")
    if dataset_name is not None and not isinstance(dataset_name, str):
        raise TypeError("dataset_name must be a string or None")
    if classification is not None and not isinstance(classification, DatasetClassification):
        raise TypeError("classification must be a DatasetClassification or None")

    if not question.strip():
        return build_controlled_response(
            "NEEDS_CLARIFICATION",
            "A non-empty factual question is required.",
            signals=["insight_service:empty_question"],
        ).model_dump(mode="json")

    active_classification = classification or classify_dataframe(
        dataframe=dataframe,
        pipeline1_report=pipeline1_report,
        dataset_name=dataset_name,
    )
    deterministic = interpret_query(
        raw_text=question,
        dataframe=dataframe,
        pipeline1_report=pipeline1_report,
        classification=active_classification,
        context="insights",
    )
    interpretation = interpret_query_with_fallback(
        question,
        dataframe,
        pipeline1_report=pipeline1_report,
        classification=active_classification,
        context="insights",
        provider=llm_provider,
        deterministic=deterministic,
    )
    adaptation = adapt_insight_query(interpretation, active_classification)
    if adaptation.status != "READY":
        return build_controlled_response(
            adaptation.status,
            adaptation.message,
            clarification_options=list(adaptation.clarification_options),
            errors=list(adaptation.errors),
            signals=[*adaptation.signals, "insight_service:stopped_before_validation"],
            failure_code=adaptation.failure_code,
        ).model_dump(mode="json")

    assert adaptation.query is not None
    validation = validate_insight_query(
        dataframe, active_classification, adaptation.query
    )
    if validation.status != "VALID":
        response_status = (
            "NEEDS_CLARIFICATION"
            if validation.status == "NEEDS_CLARIFICATION"
            else "INVALID"
        )
        return build_controlled_response(
            response_status,
            validation.message,
            clarification_options=list(validation.clarification_options),
            errors=list(validation.errors),
            signals=[*validation.signals, "insight_service:stopped_before_execution"],
            failure_code=validation.failure_code,
        ).model_dump(mode="json")

    assert validation.query is not None
    computed = execute_insight_query(dataframe, validation.query)
    response = build_insight_response(
        computed,
        signals=[*validation.signals, "insight_service:analysis_executed"],
    )
    return response.model_dump(mode="json")


__all__ = ["build_insight_results"]
