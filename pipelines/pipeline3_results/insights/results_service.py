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
from pipelines.pipeline3_results.insights.schemas import InsightResponse
from pipelines.pipeline3_results.nlp.schemas import QueryInterpretation
from pipelines.pipeline3_results.nlp.semantic_interpreter import (
    interpret_query_pretrained_first,
)
from intelligence.llm.provider import LLMProvider
from pipelines.pipeline3_results.visualization.column_classifier import (
    classify_dataframe,
)
from pipelines.pipeline3_results.visualization.schemas import DatasetClassification
from pipelines.pipeline3_results.planning.schemas import AnalyticalPlan


def build_insight_results(
    dataframe: pd.DataFrame,
    question: str,
    pipeline1_report: Mapping[str, Any] | None = None,
    dataset_name: str | None = None,
    classification: DatasetClassification | None = None,
    llm_provider: LLMProvider | None = None,
    response_provider: LLMProvider | None = None,
    interpretation: QueryInterpretation | None = None,
    analytical_plan: AnalyticalPlan | None = None,
    requested_language: str = "en",
    conversation_context: Mapping[str, Any] | None = None,
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
    if analytical_plan is not None and not isinstance(analytical_plan, AnalyticalPlan):
        raise TypeError("analytical_plan must be an AnalyticalPlan or None")

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
    active_plan = analytical_plan
    active_interpretation = interpretation
    if active_plan is None:
        active_interpretation = active_interpretation or interpret_query_pretrained_first(
            question,
            dataframe,
            pipeline1_report=pipeline1_report,
            classification=active_classification,
            context="insights",
            provider=llm_provider,
        )
        if (
            active_interpretation.query.intent != "VISUALIZATION"
            and active_interpretation.analytical_plan is not None
        ):
            active_plan = AnalyticalPlan.model_validate(
                active_interpretation.analytical_plan
            )

    if active_plan is not None:
        from pipelines.pipeline3_results.execution.compatibility import (
            verified_result_to_insight,
        )
        from pipelines.pipeline3_results.execution.pandas_executor import (
            PandasAnalyticalExecutor,
        )
        from pipelines.pipeline3_results.grounding.dataset_knowledge import (
            build_dataset_knowledge,
        )

        knowledge = build_dataset_knowledge(
            pipeline1_report,
            dataset_id=dataset_name,
            dataframe_schema={str(name): str(dtype) for name, dtype in dataframe.dtypes.items()},
        )
        if str(active_plan.operation) == "EXPLORATORY":
            from pipelines.pipeline3_results.insights.landing.service import (
                InsightsLandingService,
            )

            landing = InsightsLandingService(response_provider).build(
                dataset_name or "active_dataset",
                dataframe,
                pipeline1_report=pipeline1_report,
                dataset_name=dataset_name,
                requested_language=requested_language,
            )
            return InsightResponse(
                status="READY",
                answer=landing.summary_text,
                message="Exploratory request answered from the verified dataset summary.",
                signals=[
                    f"analytical_plan:{active_plan.plan_id}",
                    "insight_service:exploratory_summary_package",
                    *landing.signals,
                ],
            ).model_dump(mode="json")
        verified = PandasAnalyticalExecutor().execute(
            active_plan, dataframe, knowledge
        )
        shared_signals = [
            f"analytical_plan:{active_plan.plan_id}",
            f"shared_executor:{verified.execution_metadata.backend}",
            f"verified_result:{verified.result_id}",
            f"verification:{verified.verification.status}",
        ]
        if verified.status != "READY":
            message = verified.errors[0].message if verified.errors else "Shared execution did not produce a result."
            return build_controlled_response(
                "NEEDS_CLARIFICATION" if verified.status == "UNSUPPORTED" else "INVALID",
                message,
                errors=[item.message for item in verified.errors],
                signals=[*shared_signals, "insight_service:shared_execution_stopped"],
                failure_code=("UNSUPPORTED_OPERATION" if verified.status == "UNSUPPORTED" else "INVALID_COMBINATION"),
            ).model_dump(mode="json")
        from pipelines.pipeline3_results.insights.response_generation import (
            InsightsResponseGenerator,
            build_response_context,
        )

        response_context = build_response_context(
            question,
            active_plan,
            verified,
            knowledge,
            requested_language=requested_language,
            conversation_context=conversation_context,
        )
        natural_response = InsightsResponseGenerator().generate(
            response_context,
            response_provider,
            plan=active_plan,
            verified_result=verified,
        )
        response_signals = [
            *shared_signals,
            f"response_generation:{natural_response.status.casefold()}",
            f"response_language:{natural_response.language}",
            (
                "response_generation:deterministic_fallback"
                if natural_response.fallback_used
                else "response_generation:validated_model_wording"
            ),
        ]
        computed = verified_result_to_insight(active_plan, verified)
        if computed is None:
            return InsightResponse(
                status="READY",
                answer=natural_response.text,
                message="Insight explained from a verified shared result.",
                signals=[
                    *response_signals,
                    "insight_service:shared_result_consumed",
                    "insight_service:verified_result_without_legacy_shape",
                ],
            ).model_dump(mode="json")
        response = build_insight_response(
            computed,
            signals=[*response_signals, "insight_service:shared_result_consumed"],
        )
        return response.model_copy(update={
            "answer": natural_response.text,
            "message": "Insight explained from a verified shared result.",
        }).model_dump(mode="json")

    assert active_interpretation is not None
    adaptation = adapt_insight_query(active_interpretation, active_classification)
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
        signals=[
            *validation.signals,
            "execution:legacy_fallback",
            "insight_service:analysis_executed",
        ],
    )
    return response.model_dump(mode="json")


__all__ = ["build_insight_results"]
