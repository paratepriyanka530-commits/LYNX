"""Grounded suggested questions with typed-plan answerability validation."""

from __future__ import annotations

import hashlib

from pipelines.pipeline3_results.grounding.schemas import (
    AnalyticalExpression,
    DatasetKnowledge,
)
from pipelines.pipeline3_results.insights.landing.opportunities import (
    discover_opportunities,
    display_label,
    make_plan,
)
from pipelines.pipeline3_results.insights.landing.schemas import SuggestedQuestion
from pipelines.pipeline3_results.planning.schemas import AnalyticalPlan
from pipelines.pipeline3_results.planning.validator import validate_analytical_plan


_EXECUTOR_CAPABILITIES = {
    "AGGREGATE", "GROUP_AGGREGATE", "COUNT", "RANK", "TOP_N", "BOTTOM_N",
    "TREND", "DISTRIBUTION", "RELATIONSHIP", "STATISTIC", "PART_TO_WHOLE",
}


def _question_id(plan: AnalyticalPlan) -> str:
    digest = hashlib.sha256(plan.plan_id.encode("utf-8")).hexdigest()[:16]
    return f"question_{digest}"


def _wording(goal: str, labels: tuple[str, ...], language: str) -> str:
    family = language.casefold()
    if family == "hi":
        templates = {
            "AVERAGE": "{0} का औसत क्या है?",
            "RANK": "किस {0} में सबसे अधिक {1} है?",
            "COUNT_DISTINCT": "अलग-अलग {0} मान कितने हैं?",
            "TREND": "समय के साथ {0} कैसे बदला है?",
            "DISTRIBUTION": "{0} का वितरण कैसा है?",
            "RELATIONSHIP": "{0} और {1} के बीच संबंध क्या है?",
            "PART_TO_WHOLE": "कुल रिकॉर्ड में प्रत्येक {0} की हिस्सेदारी क्या है?",
        }
    elif family in {"hi-latn", "hinglish"}:
        templates = {
            "AVERAGE": "Average {0} kya hai?",
            "RANK": "Kis {0} mein sabse zyada {1} hai?",
            "COUNT_DISTINCT": "Distinct {0} values kitni hain?",
            "TREND": "Time ke saath {0} kaise badla hai?",
            "DISTRIBUTION": "{0} ka distribution kaisa hai?",
            "RELATIONSHIP": "{0} aur {1} ke beech relationship kya hai?",
            "PART_TO_WHOLE": "Total records mein har {0} ka share kya hai?",
        }
    else:
        templates = {
            "AVERAGE": "What is the average {0}?",
            "RANK": "Which {0} has the highest {1}?",
            "COUNT_DISTINCT": "How many distinct {0} values are there?",
            "TREND": "How has {0} changed over time?",
            "DISTRIBUTION": "How is {0} distributed?",
            "RELATIONSHIP": "What is the relationship between {0} and {1}?",
            "PART_TO_WHOLE": "What share of all records belongs to each {0}?",
        }
    return templates[goal].format(*labels)


def _candidate(
    goal: str,
    labels: tuple[str, ...],
    plan: AnalyticalPlan,
    knowledge: DatasetKnowledge,
    language: str,
) -> SuggestedQuestion | None:
    validation = validate_analytical_plan(plan, knowledge)
    answerable = (
        validation.status == "READY"
        and str(plan.operation) in _EXECUTOR_CAPABILITIES
    )
    if not answerable:
        return None
    return SuggestedQuestion(
        id=_question_id(plan),
        text=_wording(goal, labels, language),
        analytical_goal=goal,
        plan_template=plan,
        answerable=True,
        rationale=(
            "DatasetKnowledge identifies compatible fields and the existing plan "
            "validator confirms executor support."
        ),
    )


def generate_suggested_questions(
    knowledge: DatasetKnowledge,
    *,
    requested_language: str = "en",
    maximum: int = 5,
) -> list[SuggestedQuestion]:
    opportunities = discover_opportunities(knowledge)
    candidates: list[SuggestedQuestion] = []
    measure = opportunities.primary_measure or (
        opportunities.measures[0] if opportunities.measures else None
    )
    dimension = opportunities.primary_dimension or (
        opportunities.dimensions[0] if opportunities.dimensions else None
    )

    if measure is not None:
        plan = make_plan("AGGREGATE", measures=[AnalyticalExpression(
            kind="AGGREGATION", operation="MEAN", column=measure.name,
        )])
        item = _candidate(
            "AVERAGE", (display_label(measure),), plan, knowledge, requested_language,
        )
        if item:
            candidates.append(item)

    if dimension is not None:
        plan = make_plan(
            "RANK", measures=[AnalyticalExpression(kind="COUNT_ROWS")],
            dimension=dimension.name, comparison="HIGHEST",
        )
        item = _candidate(
            "RANK", (display_label(dimension), "record count"),
            plan, knowledge, requested_language,
        )
        if item:
            candidates.append(item)

    if len(opportunities.identifiers) == 1:
        identifier = opportunities.identifiers[0]
        plan = make_plan("COUNT", measures=[AnalyticalExpression(
            kind="COUNT_DISTINCT", operation="COUNT_DISTINCT", column=identifier.name,
        )])
        item = _candidate(
            "COUNT_DISTINCT", (display_label(identifier),),
            plan, knowledge, requested_language,
        )
        if item:
            candidates.append(item)

    if measure is not None and opportunities.temporal:
        temporal = opportunities.temporal[0]
        plan = make_plan(
            "TREND", measures=[AnalyticalExpression(
                kind="AGGREGATION", operation="MEAN", column=measure.name,
            )], temporal=temporal.name,
        )
        item = _candidate(
            "TREND", (display_label(measure),), plan, knowledge, requested_language,
        )
        if item:
            candidates.append(item)

    if measure is not None:
        plan = make_plan("DISTRIBUTION", measures=[AnalyticalExpression(
            kind="COLUMN", column=measure.name,
        )])
        item = _candidate(
            "DISTRIBUTION", (display_label(measure),),
            plan, knowledge, requested_language,
        )
        if item:
            candidates.append(item)

    if len(opportunities.measures) >= 2:
        first, second = opportunities.measures[:2]
        plan = make_plan("RELATIONSHIP", measures=[
            AnalyticalExpression(kind="COLUMN", column=first.name),
            AnalyticalExpression(kind="COLUMN", column=second.name),
        ])
        item = _candidate(
            "RELATIONSHIP", (display_label(first), display_label(second)),
            plan, knowledge, requested_language,
        )
        if item:
            candidates.append(item)

    if dimension is not None:
        plan = make_plan(
            "PART_TO_WHOLE", measures=[AnalyticalExpression(kind="COUNT_ROWS")],
            dimension=dimension.name,
        )
        item = _candidate(
            "PART_TO_WHOLE", (display_label(dimension),),
            plan, knowledge, requested_language,
        )
        if item:
            candidates.append(item)

    selected: list[SuggestedQuestion] = []
    seen_goals: set[str] = set()
    for item in candidates:
        if item.analytical_goal in seen_goals:
            continue
        selected.append(item)
        seen_goals.add(item.analytical_goal)
        if len(selected) == maximum:
            break
    return selected


__all__ = ["generate_suggested_questions"]
