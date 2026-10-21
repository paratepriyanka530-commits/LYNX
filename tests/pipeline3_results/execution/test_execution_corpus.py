from __future__ import annotations

from datetime import date
import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from pipelines.pipeline3_results.execution.pandas_executor import PandasAnalyticalExecutor
from pipelines.pipeline3_results.grounding.schemas import AnalyticalExpression, DatasetColumnKnowledge, DatasetKnowledge
from pipelines.pipeline3_results.planning.schemas import AnalyticalPlan, GroupingExpression, PlanFilter, PlanProvenance, SortExpression


CORPUS = json.loads((Path(__file__).parent / "fixtures" / "execution_corpus.json").read_text(encoding="utf-8"))


def _fixture(name: str) -> tuple[pd.DataFrame, DatasetKnowledge]:
    spec = CORPUS["datasets"][name]
    dataframe = pd.DataFrame(spec["data"])
    identifiers = set(spec.get("identifiers", []))
    categories = set(spec.get("categories", []))
    temporal = set(spec.get("temporal", []))
    columns = []
    for column, dtype in dataframe.dtypes.items():
        if column in identifiers:
            kwargs = {"semantic_type":"identifier", "semantic_role":"primary_identifier", "value_function":"entity_reference"}
        elif column in categories:
            kwargs = {"semantic_type":"categorical", "semantic_role":"category", "value_function":"classification"}
        elif column in temporal:
            kwargs = {"semantic_type":"datetime", "semantic_role":"timestamp", "value_function":"temporal"}
        else:
            kwargs = {"semantic_type":"numeric_measurement", "semantic_role":"measurement", "value_function":"measurement", "numeric_meaning":"magnitude"}
        columns.append(DatasetColumnKnowledge(name=column, physical_dtype=str(dtype), **kwargs))
    return dataframe, DatasetKnowledge(columns=columns)


def _expression(raw: list) -> AnalyticalExpression:
    kind, operation, column = raw
    return AnalyticalExpression(kind=kind, operation=operation, column=column)


def _plan(case: dict) -> AnalyticalPlan:
    raw_measures = case["measures"] if "measures" in case else [case["measure"]]
    measures = [_expression(item) for item in raw_measures]
    dimensions = [GroupingExpression(kind="GROUP_BY", column=case["dimension"])] if case.get("dimension") else []
    temporal = [GroupingExpression(kind="TIME_BUCKET", column=case["temporal"][0], time_granularity=case["temporal"][1])] if case.get("temporal") else []
    sort = []
    if case.get("sort"):
        sort = [SortExpression(target="TEMPORAL", direction=case["sort"], column=temporal[0].column)] if temporal else [SortExpression(target="MEASURE", direction=case["sort"], measure_index=0)]
    filters = []
    for column, operator, value_type, value in case.get("filters", []):
        if value_type == "DATE":
            value = date.fromisoformat(value)
        filters.append(PlanFilter(column=column, operator=operator, value_type=value_type, value=value))
    digest = hashlib.sha256(case["id"].encode()).hexdigest()[:16]
    return AnalyticalPlan(
        plan_id=f"plan_{digest}", operation=case["operation"], measures=measures,
        dimensions=dimensions, temporal=temporal, filters=filters, sort=sort,
        limit=case.get("limit"), comparison=case.get("comparison", "NONE"),
        provenance=PlanProvenance(grounding_status="RESOLVED"),
    )


@pytest.mark.parametrize("case", CORPUS["cases"], ids=lambda case: case["id"])
def test_shared_execution_corpus(case: dict) -> None:
    dataframe, knowledge = _fixture(case["dataset"])
    result = PandasAnalyticalExecutor().execute(_plan(case), dataframe, knowledge)

    assert result.status == case.get("status", "READY"), result.model_dump()
    if result.status != "READY":
        return
    assert result.result_type == case["type"]
    assert result.verification.status in {"VERIFIED", "VERIFIED_WITH_WARNINGS"}
    if "scalar" in case:
        assert math.isclose(float(result.scalar), float(case["scalar"]), rel_tol=1e-9)
    if "row_count" in case:
        assert len(result.rows) == case["row_count"]
    if "groups" in case:
        dimension = case["dimension"]
        assert [row[dimension] for row in result.rows] == case["groups"]
    if "detail" in case:
        assert result.execution_metadata.details[case["detail"][0]] == case["detail"][1]
    if "share_total" in case:
        assert math.isclose(sum(float(row["share_of_total"]) for row in result.rows), case["share_total"])
    if "excluded_invalid_min" in case:
        assert result.coverage.excluded_invalid_rows >= case["excluded_invalid_min"]
    if "excluded_null_min" in case:
        assert result.coverage.excluded_null_rows >= case["excluded_null_min"]


def test_result_identity_is_deterministic() -> None:
    case = CORPUS["cases"][0]
    dataframe, knowledge = _fixture(case["dataset"])
    executor = PandasAnalyticalExecutor()
    first = executor.execute(_plan(case), dataframe, knowledge)
    second = executor.execute(_plan(case), dataframe.copy(), knowledge)
    assert first.result_id == second.result_id


def test_execution_corpus_covers_required_domains_and_cases() -> None:
    assert len(CORPUS["cases"]) >= 30
    assert {"retail","hr","healthcare","banking","iot","cybersecurity","manufacturing","generic"}.issubset(CORPUS["datasets"])
