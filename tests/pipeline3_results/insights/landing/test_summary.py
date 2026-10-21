from __future__ import annotations

from typing import Any, Type

import pandas as pd
import pytest
from pydantic import BaseModel

from intelligence.llm.provider import LLMProvider, LLMProviderUnavailableError
from pipelines.pipeline3_results.grounding.dataset_knowledge import build_dataset_knowledge
from pipelines.pipeline3_results.insights.landing.service import InsightsLandingService
from pipelines.pipeline3_results.insights.landing.summary_engine import build_summary_package
from pipelines.pipeline3_results.insights.landing.wording import SummaryWordingService
from pipelines.pipeline3_results.insights.results_service import build_insight_results
from pipelines.pipeline3_results.planning.schemas import AnalyticalPlan, PlanProvenance


class SummaryProvider(LLMProvider):
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


def _report(
    dataframe: pd.DataFrame,
    *,
    measures: tuple[str, ...] = (), dimensions: tuple[str, ...] = (),
    temporal: tuple[str, ...] = (), identifiers: tuple[str, ...] = (),
    confidence: dict[str, float] | None = None,
    missing: dict[str, int] | None = None,
) -> dict[str, Any]:
    confidence = confidence or {}
    missing = missing or {}
    columns = []
    profiles = []
    for name in dataframe.columns:
        if name in measures:
            semantic = {"semantic_type":"numeric_measurement", "semantic_role":"measurement", "value_function":"measurement", "numeric_meaning":"magnitude"}
        elif name in dimensions:
            semantic = {"semantic_type":"categorical", "semantic_role":"category", "value_function":"classification"}
        elif name in temporal:
            semantic = {"semantic_type":"datetime", "semantic_role":"timestamp", "value_function":"temporal"}
        elif name in identifiers:
            semantic = {"semantic_type":"identifier", "semantic_role":"primary_identifier", "value_function":"entity_reference"}
        else:
            semantic = {"semantic_type":"unknown"}
        columns.append({
            "column_name":name,
            "concept_description":name.replace("_", " ").title(),
            "confidence":confidence.get(name, 0.9),
            **semantic,
        })
        profiles.append({
            "column_name":name, "physical_dtype":str(dataframe[name].dtype),
            "row_count":len(dataframe),
            "non_null_count":int(dataframe[name].notna().sum()),
            "missing":missing.get(name, int(dataframe[name].isna().sum())),
        })
    return {
        "semantic_profile":{"columns":columns, "dataset_context":{"row_entity":"record"}},
        "before_profile":{"column_profiles":profiles},
        "quality_summary":{"quality_score":0.91},
    }


def _package(dataframe: pd.DataFrame, report: dict[str, Any] | None):
    knowledge = build_dataset_knowledge(
        report,
        dataset_id="dataset",
        dataframe_schema={name:str(dtype) for name, dtype in dataframe.dtypes.items()},
    )
    return build_summary_package(dataframe, knowledge, dataset_id="dataset")


@pytest.mark.parametrize(
    ("domain", "columns"),
    [
        ("retail", {"amount":[10.0,20.0,30.0], "category":["A","A","B"], "date":["2025-01-01","2025-02-01","2025-03-01"]}),
        ("hr", {"salary":[40.0,50.0,60.0], "department":["X","Y","Y"]}),
        ("healthcare", {"duration":[2.0,4.0,6.0], "hospital":["A","A","B"]}),
        ("iot", {"temperature":[20.0,21.0,22.0], "timestamp":["2025-01-01","2025-01-02","2025-01-03"]}),
        ("cybersecurity", {"severity":[1.0,2.0,3.0], "attack_type":["x","y","x"]}),
        ("banking", {"balance":[100.0,200.0,300.0], "branch":["n","s","n"]}),
        ("manufacturing", {"cycle_time":[3.0,4.0,5.0], "line":["L1","L1","L2"]}),
        ("survey", {"score":[1.0,4.0,5.0], "segment":["a","b","b"]}),
    ],
)
def test_cross_domain_summaries_use_generic_families(domain, columns) -> None:
    dataframe = pd.DataFrame(columns)
    names = list(dataframe.columns)
    measures = (names[0],)
    temporal = tuple(name for name in names if name in {"date", "timestamp"})
    dimensions = tuple(name for name in names[1:] if name not in temporal)
    package = _package(
        dataframe,
        _report(dataframe, measures=measures, dimensions=dimensions, temporal=temporal),
    )
    assert package.status == "READY"
    assert 3 <= len(package.facts) <= 5
    assert package.facts[0].fact_type == "DATASET_OVERVIEW"
    assert all(domain not in fact.fact_id for fact in package.facts)


def test_no_temporal_column_does_not_claim_trend() -> None:
    dataframe = pd.DataFrame({"value":[1.0,2.0], "group":["a","b"]})
    package = _package(dataframe, _report(dataframe, measures=("value",), dimensions=("group",)))
    assert "TREND" not in package.opportunities
    assert all(fact.fact_type != "TREND_OPPORTUNITY" for fact in package.facts)


def test_no_meaningful_measure_omits_measure_findings() -> None:
    dataframe = pd.DataFrame({"group":["a","b","a"]})
    package = _package(dataframe, _report(dataframe, dimensions=("group",)))
    assert all(fact.fact_type not in {"MEASURE_AGGREGATE", "NUMERIC_RANGE"} for fact in package.facts)


def test_missing_pipeline1_metadata_uses_schema_without_quality_claim() -> None:
    dataframe = pd.DataFrame({"value":[1.0,2.0], "group":["a","b"]})
    package = _package(dataframe, None)
    assert all(fact.fact_type != "DATA_QUALITY" for fact in package.facts)
    assert package.facts


def test_high_missingness_is_reported_without_severity_exaggeration() -> None:
    dataframe = pd.DataFrame({"value":[1.0,None,None], "group":["a","b","c"]})
    package = _package(
        dataframe,
        _report(dataframe, measures=("value",), dimensions=("group",), missing={"value":2}),
    )
    quality = next(fact for fact in package.facts if fact.fact_type == "DATA_QUALITY")
    assert quality.values["columns_with_missing_values"] == 1
    text = SummaryWordingService().generate(package, None).text.casefold()
    assert "severe" not in text and "poor" not in text


def test_ambiguous_measure_priority_prevents_unsafe_main_measure_finding() -> None:
    dataframe = pd.DataFrame({"first":[1.0,2.0], "second":[3.0,4.0]})
    package = _package(dataframe, _report(dataframe, measures=("first", "second")))
    assert all(fact.fact_type not in {"MEASURE_AGGREGATE", "NUMERIC_RANGE"} for fact in package.facts)
    assert "RELATIONSHIP" in package.opportunities


def test_fact_selection_is_unique_diverse_and_bounded() -> None:
    dataframe = pd.DataFrame({
        "value":[1.0,2.0,3.0], "other":[4.0,5.0,6.0],
        "group":["a","a","b"], "date":["2025-01-01","2025-02-01","2025-03-01"],
        "entity_id":["x","y","z"],
    })
    report = _report(
        dataframe, measures=("value","other"), dimensions=("group",),
        temporal=("date",), identifiers=("entity_id",),
        confidence={"value":0.95,"other":0.7},
    )
    package = _package(dataframe, report)
    assert len(package.facts) <= 5
    assert len({fact.fact_id for fact in package.facts}) == len(package.facts)
    assert len({fact.fact_type for fact in package.facts}) >= 3


def test_summary_has_no_causal_claims_and_provider_failure_falls_back() -> None:
    dataframe = pd.DataFrame({"value":[1.0,2.0]})
    package = _package(dataframe, _report(dataframe, measures=("value",)))
    provider = SummaryProvider(error=LLMProviderUnavailableError("private endpoint"))
    response = SummaryWordingService().generate(package, provider)
    assert response.fallback_used is True
    assert "because" not in response.text.casefold()
    assert "private" not in response.text.casefold()


def test_invented_model_number_is_rejected() -> None:
    dataframe = pd.DataFrame({"value":[1.0,2.0]})
    package = _package(dataframe, _report(dataframe, measures=("value",)))
    provider = SummaryProvider({"text":"The dataset contains 999 records.", "language":"en"})
    response = SummaryWordingService().generate(package, provider)
    assert response.fallback_used is True
    assert "999" not in response.text


def test_invented_entity_and_unsupported_comparison_are_rejected() -> None:
    dataframe = pd.DataFrame({"value":[1.0,2.0], "group":["A","A"]})
    package = _package(
        dataframe, _report(dataframe, measures=("value",), dimensions=("group",)),
    )
    entity_provider = SummaryProvider({
        "text":"gamma has the highest Group record count at 2.", "language":"en",
    })
    assert SummaryWordingService().generate(
        package, entity_provider
    ).fallback_used is True

    scalar_package = _package(
        dataframe[["value"]], _report(dataframe[["value"]], measures=("value",)),
    )
    comparison_provider = SummaryProvider({
        "text":"Value is the highest at 2.", "language":"en",
    })
    assert SummaryWordingService().generate(
        scalar_package, comparison_provider
    ).fallback_used is True


def test_valid_hindi_summary_language_is_preserved() -> None:
    dataframe = pd.DataFrame({"value":[1.0,2.0]})
    package = _package(dataframe, _report(dataframe, measures=("value",)))
    provider = SummaryProvider({"text":"डेटासेट में 2 रिकॉर्ड हैं।", "language":"hi"})
    response = SummaryWordingService().generate(
        package, provider, requested_language="hi",
    )
    assert response.fallback_used is False
    assert response.language == "hi"


def test_failed_localized_generation_uses_honest_english_fallback() -> None:
    dataframe = pd.DataFrame({"value":[1.0,2.0]})
    package = _package(dataframe, _report(dataframe, measures=("value",)))
    response = SummaryWordingService().generate(
        package,
        SummaryProvider(error=LLMProviderUnavailableError("offline")),
        requested_language="hi",
    )
    assert response.fallback_used is True
    assert response.language == "en"


def test_provider_receives_facts_not_raw_dataframe_rows() -> None:
    dataframe = pd.DataFrame({"value":[111.12345,222.98765]})
    package = _package(dataframe, _report(dataframe, measures=("value",)))
    provider = SummaryProvider({"text":"The dataset contains 2 records across 1 columns.", "language":"en"})
    SummaryWordingService().generate(package, provider)
    prompt = provider.prompts[0]
    assert "111.12345" not in prompt
    assert "222.98765" not in prompt
    assert "dataframe" not in prompt.casefold()


def test_exploratory_plan_uses_summary_package() -> None:
    dataframe = pd.DataFrame({"value":[1.0,2.0], "group":["a","b"]})
    plan = AnalyticalPlan(
        plan_id="plan_abcdef0123456789",
        operation="EXPLORATORY",
        provenance=PlanProvenance(grounding_status="RESOLVED"),
    )
    response = build_insight_results(
        dataframe, "What stands out?", analytical_plan=plan,
    )
    assert response["status"] == "READY"
    assert response["answer"]
    assert "insight_service:exploratory_summary_package" in response["signals"]


def test_landing_cache_reuses_facts_and_complete_state() -> None:
    dataframe = pd.DataFrame({"value":[1.0,2.0]})
    report = _report(dataframe, measures=("value",))
    service = InsightsLandingService()
    first = service.build("dataset", dataframe, pipeline1_report=report)
    second = service.build("dataset", dataframe, pipeline1_report=report)
    assert first.summary == second.summary
    assert "insights_landing:cache_hit" in second.signals
