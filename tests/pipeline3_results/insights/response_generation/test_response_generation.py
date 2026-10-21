from __future__ import annotations

from typing import Any, Type

import pandas as pd
import pytest
from pydantic import BaseModel

from intelligence.llm.provider import (
    LLMProvider, LLMProviderTimeoutError, LLMProviderUnavailableError,
)
from pipelines.pipeline3_results.execution.schemas import (
    ExecutionCoverage, ExecutionMetadata, ResultField, ResultSchema, VerifiedResult,
)
from pipelines.pipeline3_results.grounding.schemas import (
    AnalyticalExpression, DatasetColumnKnowledge, DatasetKnowledge,
)
from pipelines.pipeline3_results.insights.response_generation import (
    InsightsResponseGenerator, build_response_context,
)
from pipelines.pipeline3_results.insights.response_generation.context_builder import (
    MAX_RESPONSE_ROWS, format_number, serialize_verified_result,
)
from pipelines.pipeline3_results.insights.response_generation.prompt import (
    SYSTEM_PROMPT, build_response_prompt,
)
from pipelines.pipeline3_results.insights.response_generation.schemas import ModelResponseDraft
from pipelines.pipeline3_results.insights.response_generation.validator import validate_response
from pipelines.pipeline3_results.insights.results_service import build_insight_results
from pipelines.pipeline3_results.planning.schemas import (
    AnalyticalPlan, GroupingExpression, PlanProvenance, SortExpression,
)


class MockProvider(LLMProvider):
    def __init__(self, response: Any = None, error: Exception | None = None):
        self.response, self.error = response, error
        self.calls = 0
        self.prompts: list[str] = []
        self.models: list[Type[BaseModel]] = []

    def generate_structured(self, prompt: str, response_model: Type[BaseModel]):
        self.calls += 1
        self.prompts.append(prompt)
        self.models.append(response_model)
        if self.error:
            raise self.error
        return self.response


def _plan(
    operation: str = "AGGREGATE", *, kind: str = "AGGREGATION",
    aggregate: str | None = "SUM", measure: str | None = "revenue",
    dimension: str | None = None, comparison: str = "NONE",
    limit: int | None = None, temporal: str | None = None,
) -> AnalyticalPlan:
    return AnalyticalPlan(
        plan_id="plan_0123456789abcdef", operation=operation,
        measures=[AnalyticalExpression(kind=kind, operation=aggregate, column=measure)],
        dimensions=[GroupingExpression(kind="GROUP_BY", column=dimension)] if dimension else [],
        temporal=[GroupingExpression(kind="TIME_BUCKET", column=temporal, time_granularity="MONTH")] if temporal else [],
        sort=[SortExpression(target="MEASURE", direction="DESC", measure_index=0)] if operation in {"RANK", "TOP_N", "BOTTOM_N"} else [],
        limit=limit, comparison=comparison,
        provenance=PlanProvenance(grounding_status="RESOLVED"),
    )


def _result(
    result_type: str = "SCALAR", *, scalar: int | float | None = 125430.5,
    rows: list[dict[str, Any]] | None = None, status: str = "READY",
    operation: str = "AGGREGATE", usable: int = 10, excluded: int = 0,
    details: dict[str, Any] | None = None, warnings: list[str] | None = None,
) -> VerifiedResult:
    return VerifiedResult(
        result_id="result_0123456789abcdef", plan_id="plan_0123456789abcdef",
        status=status, result_type=result_type,
        scalar=scalar if result_type in {"SCALAR", "STATISTIC"} else None,
        rows=rows or [],
        schema=ResultSchema(fields=[ResultField(name="value", dtype="number", role="VALUE")]),
        coverage=ExecutionCoverage(
            source_rows=usable + excluded, filtered_rows=usable + excluded,
            usable_rows=usable, excluded_null_rows=excluded, excluded_invalid_rows=0,
        ),
        execution_metadata=ExecutionMetadata(
            backend="pandas", operation=operation, details=details or {},
        ),
        warnings=warnings or [],
    )


def _knowledge(*columns: str) -> DatasetKnowledge:
    return DatasetKnowledge(columns=[DatasetColumnKnowledge(
        name=column,
        physical_dtype="float64" if column != "category" else "object",
        concept_description=column.replace("_", " ").title(),
    ) for column in columns])


def _context(plan: AnalyticalPlan, result: VerifiedResult, language: str = "en"):
    return build_response_context(
        "What is the requested result?", plan, result,
        _knowledge("revenue", "salary", "temperature", "category", "date", "price", "quantity"),
        requested_language=language,
    )


@pytest.mark.parametrize(
    ("plan", "result", "expected"),
    [
        (_plan(), _result(), "The total Revenue is 125,430.50."),
        (_plan(aggregate="MEAN", measure="salary"), _result(scalar=72.5), "The average Salary is 72.50."),
        (_plan("COUNT", kind="COUNT_ROWS", aggregate=None, measure=None), _result(scalar=1000, operation="COUNT"), "There are 1,000 records."),
        (_plan("COUNT", kind="COUNT_DISTINCT", aggregate="COUNT_DISTINCT", measure="category"), _result(scalar=842, operation="COUNT"), "There are 842 distinct Category values."),
        (_plan("COUNT", kind="COUNT_NON_NULL", aggregate=None, measure="salary"), _result(scalar=914, operation="COUNT"), "Salary is available for 914 records."),
        (_plan("STATISTIC", kind="STATISTIC", aggregate="STDDEV", measure="temperature"), _result("STATISTIC", scalar=4.21, operation="STATISTIC", details={"definition":"sample standard deviation (ddof=1)"}), "The sample standard deviation of Temperature is 4.21."),
        (_plan("STATISTIC", kind="STATISTIC", aggregate="RANGE", measure="temperature"), _result("STATISTIC", scalar=18.6, operation="STATISTIC", details={"definition":"maximum minus minimum"}), "The Temperature range is 18.60."),
    ],
)
def test_scalar_and_count_fallback_semantics(plan, result, expected) -> None:
    response = InsightsResponseGenerator().generate(
        _context(plan, result), None, plan=plan, verified_result=result,
    )
    assert response.text == expected
    assert response.fallback_used is True


def test_grouped_highest_tie_and_top_n_wording() -> None:
    rows = [
        {"category":"Hospital A", "value":542.0},
        {"category":"Hospital B", "value":542.0},
    ]
    rank = _plan("RANK", dimension="category", comparison="HIGHEST")
    result = _result("TABLE", rows=rows, scalar=None, operation="RANK")
    answer = InsightsResponseGenerator().generate(
        _context(rank, result), None, plan=rank, verified_result=result,
    )
    assert "Hospital A and Hospital B are tied" in answer.text
    assert "542.00 each" in answer.text

    top = _plan("TOP_N", dimension="category", comparison="TOP_N", limit=2)
    top_answer = InsightsResponseGenerator().generate(
        _context(top, result), None, plan=top, verified_result=result,
    )
    assert top_answer.text.startswith("The top 2 results")


def test_grouped_trend_distribution_relationship_and_part_to_whole() -> None:
    grouped_plan = _plan("GROUP_AGGREGATE", dimension="category")
    grouped_result = _result(
        "TABLE", scalar=None, rows=[{"category":"A", "value":120.0}],
        operation="GROUP_AGGREGATE",
    )
    grouped = InsightsResponseGenerator().generate(
        _context(grouped_plan, grouped_result), None,
        plan=grouped_plan, verified_result=grouped_result,
    )
    assert "A (120.00)" in grouped.text

    trend_plan = _plan("TREND", temporal="date")
    trend_result = _result(
        "SERIES", scalar=None,
        rows=[
            {"date":"2025-01-01T00:00:00", "value":10.0},
            {"date":"2025-02-01T00:00:00", "value":15.0},
        ], operation="TREND",
    )
    trend = InsightsResponseGenerator().generate(
        _context(trend_plan, trend_result), None,
        plan=trend_plan, verified_result=trend_result,
    )
    assert "starts at 10.00" in trend.text

    distribution_plan = _plan(
        "DISTRIBUTION", kind="COLUMN", aggregate=None, measure="salary",
    )
    distribution_result = _result(
        "DISTRIBUTION", scalar=None,
        rows=[
            {"bin_start":0.0,"bin_end":10.0,"count":4},
            {"bin_start":10.0,"bin_end":20.0,"count":8},
        ], operation="DISTRIBUTION",
    )
    distribution = InsightsResponseGenerator().generate(
        _context(distribution_plan, distribution_result), None,
        plan=distribution_plan, verified_result=distribution_result,
    )
    assert "10.00 to 20.00" in distribution.text

    relationship_plan = _plan(
        "RELATIONSHIP", kind="COLUMN", aggregate=None, measure="price",
    ).model_copy(update={"measures":[
        AnalyticalExpression(kind="COLUMN", column="price"),
        AnalyticalExpression(kind="COLUMN", column="quantity"),
    ]})
    relationship_result = _result(
        "RELATIONSHIP", scalar=None,
        rows=[{"price":1.0,"quantity":2.0}],
        operation="RELATIONSHIP", usable=420,
    )
    relationship = InsightsResponseGenerator().generate(
        _context(relationship_plan, relationship_result), None,
        plan=relationship_plan, verified_result=relationship_result,
    )
    assert "420 valid Price–Quantity pairs" in relationship.text
    assert "correlat" not in relationship.text.casefold()

    share_plan = _plan("PART_TO_WHOLE", dimension="category")
    share_result = _result(
        "TABLE", scalar=None,
        rows=[{"category":"Veggie", "value":423.0, "share_of_total":0.423}],
        operation="PART_TO_WHOLE",
    )
    share = InsightsResponseGenerator().generate(
        _context(share_plan, share_result), None,
        plan=share_plan, verified_result=share_result,
    )
    assert "42.3%" in share.text


def test_empty_partial_and_language_handling() -> None:
    plan = _plan()
    empty = _result("TABLE", scalar=None, status="EMPTY", usable=0)
    response = InsightsResponseGenerator().generate(
        _context(plan, empty), None, plan=plan, verified_result=empty,
    )
    assert response.text.startswith("No matching records")

    partial = _result(status="PARTIAL", excluded=108, warnings=["Coverage is partial."])
    partial_response = InsightsResponseGenerator().generate(
        _context(plan, partial), None, plan=plan, verified_result=partial,
    )
    assert "108 records were excluded" in partial_response.text

    for language in ("en", "hi", "hi-Latn"):
        localized = InsightsResponseGenerator().generate(
            _context(plan, _result(), language), None,
            plan=plan, verified_result=_result(),
        )
        assert localized.language == language


@pytest.mark.parametrize(
    ("text", "error_fragment"),
    [
        ("The total Revenue is 999.00.", "unsupported numeric"),
        ("The total Revenue is 125,431.50.", "unsupported numeric"),
        ("Veggie contributes 51%.", "unsupported numeric"),
        ("Gamma has the highest Revenue at 125,430.50.", "unsupported factual entity"),
        ("gamma has the highest Revenue at 125,430.50.", "unsupported factual entity claim"),
        ("Revenue increased because demand improved.", "unsupported causal"),
        ("Price and Quantity are strongly correlated.", "unsupported statistical"),
        ("The pandas dataframe result is 125,430.50.", "internal implementation"),
        ("The total Revenue is $125,430.50.", "unsupported currency unit"),
    ],
)
def test_validator_rejects_fabricated_or_internal_claims(text, error_fragment) -> None:
    validation = validate_response(
        ModelResponseDraft(text=text, language="en"), _context(_plan(), _result()),
    )
    assert validation.valid is False
    assert any(error_fragment in error for error in validation.errors)


def test_valid_model_wording_is_used_without_recalculation() -> None:
    plan, result = _plan(), _result()
    provider = MockProvider({
        "text":"The total Revenue is 125,430.50.", "language":"en",
    })
    response = InsightsResponseGenerator().generate(
        _context(plan, result), provider, plan=plan, verified_result=result,
    )
    assert response.text == "The total Revenue is 125,430.50."
    assert response.fallback_used is False
    assert provider.models == [ModelResponseDraft]


def test_wrong_model_language_falls_back_to_requested_language() -> None:
    plan, result = _plan(), _result()
    provider = MockProvider({
        "text":"The total Revenue is 125,430.50.", "language":"en",
    })
    response = InsightsResponseGenerator().generate(
        _context(plan, result, "hi"), provider,
        plan=plan, verified_result=result,
    )
    assert response.fallback_used is True
    assert response.language == "hi"


@pytest.mark.parametrize(
    "provider",
    [
        MockProvider("not a mapping"),
        MockProvider(error=LLMProviderUnavailableError("secret endpoint")),
        MockProvider(error=LLMProviderTimeoutError("secret timeout")),
        MockProvider(error=RuntimeError("private provider failure")),
    ],
)
def test_malformed_or_failed_provider_uses_deterministic_fallback(provider) -> None:
    plan, result = _plan(), _result()
    response = InsightsResponseGenerator().generate(
        _context(plan, result), provider, plan=plan, verified_result=result,
    )
    assert response.fallback_used is True
    assert response.text == "The total Revenue is 125,430.50."
    assert "secret" not in response.text


def test_context_is_bounded_and_contains_no_dataframe_or_raw_report() -> None:
    plan = _plan("GROUP_AGGREGATE", dimension="category")
    rows = [{"category":f"C{index}", "value":float(index)} for index in range(100)]
    result = _result(
        "TABLE", scalar=None, rows=rows,
        operation="GROUP_AGGREGATE", usable=100,
    )
    context = _context(plan, result)
    serialized, truncated = serialize_verified_result(plan, result)
    assert truncated is True
    assert len(serialized["rows"]) == MAX_RESPONSE_ROWS
    prompt = build_response_prompt(context).casefold()
    assert "raw dataframe" not in prompt
    assert "pipeline1_report" not in prompt
    assert "c99" not in prompt
    assert "chain-of-thought" in SYSTEM_PROMPT


def test_service_integration_uses_validated_answer_and_preserves_shape() -> None:
    dataframe = pd.DataFrame({"revenue":[10.0, 20.0, 30.0]})
    provider = MockProvider({"text":"The total revenue is 60.00.", "language":"en"})
    response = build_insight_results(
        dataframe, "What is total revenue?",
        analytical_plan=_plan(), response_provider=provider,
    )
    assert response["status"] == "READY"
    assert response["answer"] == "The total revenue is 60.00."
    assert "response_generation:validated_model_wording" in response["signals"]
    assert set(response) == {
        "status", "answer", "result", "evidence", "message",
        "clarification_options", "errors", "signals", "failure_code",
    }


def test_clarification_bypasses_response_generation() -> None:
    dataframe = pd.DataFrame({"gross_sales":[1, 2], "net_sales":[1, 2]})
    provider = MockProvider({"text":"Invented answer 3.", "language":"en"})
    response = build_insight_results(
        dataframe, "What is total sales?", response_provider=provider,
    )
    assert response["status"] == "NEEDS_CLARIFICATION"
    assert provider.calls == 0


@pytest.mark.parametrize(
    "domain",
    [
        "retail", "HR", "healthcare", "banking", "IoT", "cybersecurity",
        "manufacturing", "generic business",
    ],
)
def test_cross_domain_generation_uses_one_generic_contract(domain: str) -> None:
    plan, result = _plan(), _result()
    context = build_response_context(
        f"Give the {domain} total revenue", plan, result, _knowledge("revenue"),
    )
    provider = MockProvider({
        "text":"The total Revenue is 125,430.50.", "language":"en",
    })
    response = InsightsResponseGenerator().generate(
        context, provider, plan=plan, verified_result=result,
    )
    assert response.fallback_used is False


def test_formatting_preserves_precision_and_adds_no_unit() -> None:
    assert format_number(125430.5) == "125,430.50"
    assert format_number(4.21001) == "4.21001"
    assert _context(_plan(), _result()).units_or_formats_if_known == {}
