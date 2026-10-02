"""Phase 10 deterministic structured-conversation coverage."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from pipelines.pipeline3_results.conversation.context_store import (
    ConversationDatasetMismatchError,
    ConversationNotFoundError,
)
from pipelines.pipeline3_results.conversation.orchestrator import (
    ConversationOrchestrator,
)


@pytest.fixture
def dataframe() -> pd.DataFrame:
    return pd.DataFrame({
        "sales": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "revenue": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "profit": [1.0, 3.0, 2.0, 8.0, 5.0, 9.0],
        "salary": [50.0, 70.0, 60.0, 80.0, 90.0, 100.0],
        "region": ["North", "North", "South", "South", "East", "East"],
        "department": ["A", "A", "B", "B", "C", "C"],
        "category": ["X", "Y", "X", "Y", "X", "Y"],
        "product": ["P1", "P2", "P3", "P4", "P5", "P6"],
        "event_date": pd.to_datetime([
            "2024-01-01", "2024-02-01", "2025-01-01",
            "2025-02-01", "2026-01-01", "2026-02-01",
        ]),
    })


def test_new_visualization_conversation(dataframe) -> None:
    result = ConversationOrchestrator().start(
        "dataset-a", dataframe, "show sales by region"
    )
    assert result.intent == "VISUALIZATION"
    assert result.status == "READY"
    assert result.visualization is not None
    assert result.insight is None


def test_new_insight_conversation(dataframe) -> None:
    result = ConversationOrchestrator().start(
        "dataset-a", dataframe, "which region has highest sales?"
    )
    assert result.intent == "INSIGHT"
    assert result.status == "READY"
    assert result.insight["result"]["winner"] == "East"


def test_aggregation_followup_preserves_fields(dataframe) -> None:
    orchestrator = ConversationOrchestrator()
    first = orchestrator.start("dataset-a", dataframe, "show salary by department")
    second = orchestrator.continue_conversation(
        "dataset-a", first.conversation_id, dataframe, "use average instead"
    )
    state = orchestrator.store.get("dataset-a", first.conversation_id).active_query
    assert second.status == "READY"
    assert state.measure == "salary"
    assert state.dimension == "department"
    assert state.aggregation == "MEAN"


def test_dimension_replacement_preserves_measure(dataframe) -> None:
    orchestrator = ConversationOrchestrator()
    first = orchestrator.start("dataset-a", dataframe, "show revenue by region")
    orchestrator.continue_conversation(
        "dataset-a", first.conversation_id, dataframe, "by category instead"
    )
    state = orchestrator.store.get("dataset-a", first.conversation_id).active_query
    assert state.measure == "revenue"
    assert state.dimension == "category"


def test_measure_replacement_preserves_dimension(dataframe) -> None:
    orchestrator = ConversationOrchestrator()
    first = orchestrator.start("dataset-a", dataframe, "show sales by region")
    orchestrator.continue_conversation(
        "dataset-a", first.conversation_id, dataframe, "use profit instead"
    )
    state = orchestrator.store.get("dataset-a", first.conversation_id).active_query
    assert state.measure == "profit"
    assert state.dimension == "region"


def test_comparison_replacement(dataframe) -> None:
    orchestrator = ConversationOrchestrator()
    first = orchestrator.start(
        "dataset-a", dataframe, "which department has highest salary?"
    )
    second = orchestrator.continue_conversation(
        "dataset-a", first.conversation_id, dataframe, "lowest?"
    )
    state = orchestrator.store.get("dataset-a", first.conversation_id).active_query
    assert second.intent == "INSIGHT"
    assert state.comparison == "LOWEST"
    assert second.insight["result"]["winner"] == "A"


def test_top_bottom_limit_modification(dataframe) -> None:
    orchestrator = ConversationOrchestrator()
    first = orchestrator.start(
        "dataset-a", dataframe, "top 5 products by revenue"
    )
    assert first.status == "READY"
    second = orchestrator.continue_conversation(
        "dataset-a", first.conversation_id, dataframe, "show bottom 3 instead"
    )
    state = orchestrator.store.get("dataset-a", first.conversation_id).active_query
    assert state.comparison == "BOTTOM_N"
    assert state.limit == 3
    assert len(second.insight["result"]["rows"]) == 3


def test_clarification_continuation_selects_choice() -> None:
    dataframe = pd.DataFrame({
        "gross_sales": [10.0, 20.0, 30.0, 40.0],
        "net_sales": [8.0, 16.0, 24.0, 32.0],
        "region": ["N", "S", "N", "S"],
    })
    orchestrator = ConversationOrchestrator()
    first = orchestrator.start("dataset-a", dataframe, "show sales by region")
    assert first.status == "NEEDS_CLARIFICATION"
    second = orchestrator.continue_conversation(
        "dataset-a", first.conversation_id, dataframe, "net sales"
    )
    assert second.status == "READY"
    assert second.visualization["visualizations"][0]["y_column"] == "net_sales"


def test_wrong_clarification_answer_keeps_pending() -> None:
    dataframe = pd.DataFrame({
        "gross_sales": [10.0, 20.0], "net_sales": [8.0, 16.0],
        "region": ["N", "S"],
    })
    orchestrator = ConversationOrchestrator()
    first = orchestrator.start("dataset-a", dataframe, "show sales by region")
    second = orchestrator.continue_conversation(
        "dataset-a", first.conversation_id, dataframe, "profit"
    )
    context = orchestrator.store.get("dataset-a", first.conversation_id)
    assert second.status == "NEEDS_CLARIFICATION"
    assert second.clarification_options == ["gross_sales", "net_sales"]
    assert context.pending_clarification is not None


def test_visualization_to_insight_explicit_switch(dataframe) -> None:
    orchestrator = ConversationOrchestrator()
    first = orchestrator.start("dataset-a", dataframe, "show revenue by region")
    second = orchestrator.continue_conversation(
        "dataset-a",
        first.conversation_id,
        dataframe,
        "Which region actually has the highest revenue?",
    )
    assert second.intent == "INSIGHT"
    assert second.insight is not None
    assert second.visualization is None


def test_insight_to_visualization_explicit_switch(dataframe) -> None:
    orchestrator = ConversationOrchestrator()
    first = orchestrator.start(
        "dataset-a", dataframe, "Which region has highest revenue?"
    )
    second = orchestrator.continue_conversation(
        "dataset-a", first.conversation_id, dataframe, "show revenue by region"
    )
    assert second.intent == "VISUALIZATION"
    assert second.visualization is not None
    assert second.insight is None


def test_invalid_chart_change_is_revalidated(dataframe) -> None:
    orchestrator = ConversationOrchestrator()
    first = orchestrator.start("dataset-a", dataframe, "show salary by department")
    second = orchestrator.continue_conversation(
        "dataset-a", first.conversation_id, dataframe, "make it a scatter plot"
    )
    assert second.status == "INVALID"
    assert second.visualization["status"] == "INVALID"


def test_filter_refinement_is_structured_and_controlled(dataframe) -> None:
    orchestrator = ConversationOrchestrator()
    first = orchestrator.start("dataset-a", dataframe, "show sales by region")
    second = orchestrator.continue_conversation(
        "dataset-a", first.conversation_id, dataframe, "only North"
    )
    state = orchestrator.store.get("dataset-a", first.conversation_id).active_query
    assert state.filters[0].resolved_column == "region"
    assert state.filters[0].value_term == "north"
    assert second.status == "NEEDS_CLARIFICATION"  # existing engine safely declines filters


def test_temporal_refinement_is_structured_and_controlled(dataframe) -> None:
    orchestrator = ConversationOrchestrator()
    first = orchestrator.start("dataset-a", dataframe, "show sales by region")
    second = orchestrator.continue_conversation(
        "dataset-a", first.conversation_id, dataframe, "only for 2025"
    )
    state = orchestrator.store.get("dataset-a", first.conversation_id).active_query
    assert state.temporal_terms == ("in 2025",)
    # The current single-turn validator rejects unresolved calendar restrictions;
    # conversation preserves the refinement but never bypasses that authority.
    assert second.status == "INVALID"


def test_conversation_cannot_cross_datasets(dataframe) -> None:
    orchestrator = ConversationOrchestrator()
    first = orchestrator.start("dataset-a", dataframe, "show sales by region")
    with pytest.raises(ConversationDatasetMismatchError):
        orchestrator.continue_conversation(
            "dataset-b", first.conversation_id, dataframe, "use average instead"
        )


def test_unknown_conversation_id(dataframe) -> None:
    orchestrator = ConversationOrchestrator()
    with pytest.raises(ConversationNotFoundError):
        orchestrator.continue_conversation(
            "dataset-a", "unknown", dataframe, "use average instead"
        )


def test_clear_conversation(dataframe) -> None:
    orchestrator = ConversationOrchestrator()
    first = orchestrator.start("dataset-a", dataframe, "show sales by region")
    orchestrator.clear("dataset-a", first.conversation_id)
    with pytest.raises(ConversationNotFoundError):
        orchestrator.store.get("dataset-a", first.conversation_id)


def test_dataframe_is_not_mutated(dataframe) -> None:
    before = dataframe.copy(deep=True)
    orchestrator = ConversationOrchestrator()
    first = orchestrator.start("dataset-a", dataframe, "show sales by region")
    orchestrator.continue_conversation(
        "dataset-a", first.conversation_id, dataframe, "use average instead"
    )
    pd.testing.assert_frame_equal(dataframe, before)


def test_deterministic_repeatability_ignoring_generated_ids(dataframe) -> None:
    outcomes = []
    for _ in range(2):
        orchestrator = ConversationOrchestrator()
        first = orchestrator.start("dataset-a", dataframe, "show sales by region")
        result = orchestrator.continue_conversation(
            "dataset-a", first.conversation_id, dataframe, "use average instead"
        )
        payload = result.model_dump(mode="json")
        payload.pop("conversation_id")
        payload.pop("turn_id")
        outcomes.append(payload)
    assert outcomes[0] == outcomes[1]


def test_result_is_json_serializable(dataframe) -> None:
    result = ConversationOrchestrator().start(
        "dataset-a", dataframe, "show sales by region"
    )
    payload = result.model_dump(mode="json")
    assert json.loads(json.dumps(payload)) == payload


def test_state_contains_no_dataframe_contents(dataframe) -> None:
    orchestrator = ConversationOrchestrator()
    result = orchestrator.start("dataset-a", dataframe, "show sales by region")
    context = orchestrator.store.get("dataset-a", result.conversation_id)
    payload = context.model_dump(mode="json")
    assert "dataframe" not in json.dumps(payload).casefold()
    assert "East" not in json.dumps(payload)


def test_independent_conversations_remain_isolated(dataframe) -> None:
    orchestrator = ConversationOrchestrator()
    first = orchestrator.start("dataset-a", dataframe, "show sales by region")
    second = orchestrator.start("dataset-a", dataframe, "show profit by category")
    orchestrator.continue_conversation(
        "dataset-a", first.conversation_id, dataframe, "use average instead"
    )
    first_state = orchestrator.store.get("dataset-a", first.conversation_id).active_query
    second_state = orchestrator.store.get("dataset-a", second.conversation_id).active_query
    assert first_state.aggregation == "MEAN"
    assert second_state.measure == "profit"
    assert second_state.dimension == "category"
    assert second_state.aggregation == "SUM"
