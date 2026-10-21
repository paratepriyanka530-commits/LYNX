from __future__ import annotations

import pandas as pd
import pytest

from pipelines.pipeline3_results.grounding.dataset_knowledge import build_dataset_knowledge
from pipelines.pipeline3_results.insights.landing.suggestions import generate_suggested_questions
from pipelines.pipeline3_results.planning.validator import validate_analytical_plan
from tests.pipeline3_results.insights.landing.test_summary import _report


def _questions(
    dataframe: pd.DataFrame,
    *,
    measures: tuple[str, ...] = (), dimensions: tuple[str, ...] = (),
    temporal: tuple[str, ...] = (), identifiers: tuple[str, ...] = (),
    language: str = "en",
):
    report = _report(
        dataframe, measures=measures, dimensions=dimensions,
        temporal=temporal, identifiers=identifiers,
    )
    knowledge = build_dataset_knowledge(
        report,
        dataframe_schema={name:str(dtype) for name, dtype in dataframe.dtypes.items()},
    )
    return generate_suggested_questions(
        knowledge, requested_language=language,
    ), knowledge


def test_every_question_has_a_valid_executor_plan() -> None:
    dataframe = pd.DataFrame({
        "value":[1.0,2.0,3.0], "other":[3.0,4.0,5.0],
        "group":["a","b","a"], "date":["2025-01-01","2025-02-01","2025-03-01"],
        "entity_id":["x","y","z"],
    })
    questions, knowledge = _questions(
        dataframe, measures=("value","other"), dimensions=("group",),
        temporal=("date",), identifiers=("entity_id",),
    )
    assert questions
    assert all(item.answerable for item in questions)
    assert all(validate_analytical_plan(item.plan_template, knowledge).status == "READY" for item in questions)


def test_questions_never_reference_unsupported_columns() -> None:
    dataframe = pd.DataFrame({"amount":[1.0,2.0], "category":["a","b"]})
    questions, _ = _questions(dataframe, measures=("amount",), dimensions=("category",))
    serialized = " ".join(item.plan_template.model_dump_json() for item in questions)
    assert "salary" not in serialized
    assert "region" not in serialized


def test_no_temporal_or_relationship_question_without_compatible_fields() -> None:
    dataframe = pd.DataFrame({"value":[1.0,2.0], "group":["a","b"]})
    questions, _ = _questions(dataframe, measures=("value",), dimensions=("group",))
    goals = {item.analytical_goal for item in questions}
    assert "TREND" not in goals
    assert "RELATIONSHIP" not in goals


def test_no_measure_or_temporal_invention_for_categorical_dataset() -> None:
    dataframe = pd.DataFrame({"group":["a","b"]})
    questions, _ = _questions(dataframe, dimensions=("group",))
    goals = {item.analytical_goal for item in questions}
    assert "AVERAGE" not in goals
    assert "DISTRIBUTION" not in goals
    assert "TREND" not in goals


def test_distinct_entity_count_requires_one_identifier() -> None:
    dataframe = pd.DataFrame({"customer_id":["a","b","a"], "value":[1.0,2.0,3.0]})
    questions, _ = _questions(dataframe, measures=("value",), identifiers=("customer_id",))
    distinct = [item for item in questions if item.analytical_goal == "COUNT_DISTINCT"]
    assert len(distinct) == 1
    assert distinct[0].plan_template.measures[0].kind == "COUNT_DISTINCT"


def test_multiple_identifiers_avoid_ambiguous_distinct_question() -> None:
    dataframe = pd.DataFrame({"customer_id":["a","b"], "order_id":["x","y"]})
    questions, _ = _questions(dataframe, identifiers=("customer_id", "order_id"))
    assert all(item.analytical_goal != "COUNT_DISTINCT" for item in questions)


def test_question_goals_are_diverse_and_bounded() -> None:
    dataframe = pd.DataFrame({
        "first":[1.0,2.0], "second":[3.0,4.0], "group":["a","b"],
        "date":["2025-01-01","2025-02-01"], "id":["x","y"],
    })
    questions, _ = _questions(
        dataframe, measures=("first","second"), dimensions=("group",),
        temporal=("date",), identifiers=("id",),
    )
    goals = [item.analytical_goal for item in questions]
    assert len(questions) <= 5
    assert len(goals) == len(set(goals))
    assert len(goals) >= 3


@pytest.mark.parametrize(
    ("language", "fragment"),
    [("en", "What"), ("hi", "क्या"), ("hi-Latn", "kya")],
)
def test_question_language_templates_preserve_plan_meaning(language, fragment) -> None:
    dataframe = pd.DataFrame({"value":[1.0,2.0]})
    questions, _ = _questions(dataframe, measures=("value",), language=language)
    assert fragment.casefold() in questions[0].text.casefold()
    assert questions[0].plan_template.measures[0].column == "value"


@pytest.mark.parametrize(
    "domain",
    ["retail", "hr", "healthcare", "banking", "iot", "cybersecurity", "manufacturing", "generic"],
)
def test_cross_domain_generation_has_no_domain_specific_templates(domain) -> None:
    dataframe = pd.DataFrame({"metric":[1.0,2.0], "segment":["a","b"]})
    questions, _ = _questions(dataframe, measures=("metric",), dimensions=("segment",))
    assert questions
    assert all(domain not in item.rationale.casefold() for item in questions)


def test_only_structural_knowledge_enters_question_text_not_values() -> None:
    dataframe = pd.DataFrame({"metric":[987654.321,123456.789], "segment":["secret-a","secret-b"]})
    questions, _ = _questions(dataframe, measures=("metric",), dimensions=("segment",))
    text = " ".join(item.text for item in questions)
    assert "987654" not in text
    assert "secret-a" not in text
