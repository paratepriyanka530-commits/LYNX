"""Deterministic conversation orchestration over existing Pipeline 3 services."""

from __future__ import annotations

from typing import Any, Mapping
from uuid import uuid4

import pandas as pd

from intelligence.llm.provider import LLMProvider

from pipelines.pipeline3_results.conversation.context_store import (
    InMemoryConversationStore,
)
from pipelines.pipeline3_results.conversation.followup_resolver import (
    compile_structured_request,
    resolve_followup,
)
from pipelines.pipeline3_results.conversation.schemas import (
    ConversationContext,
    ConversationQueryState,
    ConversationResult,
    ConversationTurn,
    FollowupResolution,
    PendingClarification,
)
from pipelines.pipeline3_results.insights.results_service import build_insight_results
from pipelines.pipeline3_results.nlp.normalizer import normalize_text
from pipelines.pipeline3_results.nlp.llm.schemas import LLMFollowupInterpretation
from pipelines.pipeline3_results.nlp.llm.semantic_interpreter import (
    fallback_reason,
    interpret_followup_with_llm,
    interpret_query_with_fallback,
)
from pipelines.pipeline3_results.nlp.orchestrator import interpret_query
from pipelines.pipeline3_results.nlp.schemas import QueryFilter, QueryInterpretation
from pipelines.pipeline3_results.visualization.column_classifier import (
    classify_dataframe,
)
from pipelines.pipeline3_results.visualization.results_service import (
    build_visualization_results,
)


class ConversationOrchestrator:
    """Own context flow while leaving all analysis to existing services."""

    def __init__(
        self,
        store: InMemoryConversationStore | None = None,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        self.store = store or InMemoryConversationStore()
        self.llm_provider = llm_provider

    def start(
        self,
        dataset_id: str,
        dataframe: pd.DataFrame,
        user_text: str,
        *,
        pipeline1_report: Mapping[str, Any] | None = None,
        dataset_name: str | None = None,
    ) -> ConversationResult:
        context = self.store.create(dataset_id)
        return self._process(
            context, dataframe, user_text,
            pipeline1_report=pipeline1_report, dataset_name=dataset_name,
        )

    def continue_conversation(
        self,
        dataset_id: str,
        conversation_id: str,
        dataframe: pd.DataFrame,
        user_text: str,
        *,
        pipeline1_report: Mapping[str, Any] | None = None,
        dataset_name: str | None = None,
    ) -> ConversationResult:
        context = self.store.get(dataset_id, conversation_id)
        return self._process(
            context, dataframe, user_text,
            pipeline1_report=pipeline1_report, dataset_name=dataset_name,
        )

    def clear(self, dataset_id: str, conversation_id: str) -> None:
        self.store.clear(dataset_id, conversation_id)

    def _process(
        self,
        context: ConversationContext,
        dataframe: pd.DataFrame,
        user_text: str,
        *,
        pipeline1_report: Mapping[str, Any] | None,
        dataset_name: str | None,
    ) -> ConversationResult:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame")
        if not isinstance(user_text, str) or not user_text.strip():
            raise ValueError("user_text must be a non-empty string")

        classification = classify_dataframe(
            dataframe=dataframe,
            pipeline1_report=pipeline1_report,
            dataset_name=dataset_name,
        )
        previous_intent_context = None
        if context.active_query is not None:
            if context.active_query.intent == "VISUALIZATION":
                previous_intent_context = "visualization"
            elif context.active_query.intent == "INSIGHT":
                previous_intent_context = "insights"
        current_interpretation: QueryInterpretation | None = None

        pending = context.pending_clarification
        if pending is not None:
            selected = self._select_clarification(user_text, pending)
            if selected is None:
                return self._record_controlled_clarification(context, user_text, pending)
            state = self._apply_clarification(pending, selected)
            request_text = compile_structured_request(state)
            inherited_signals = [
                "conversation:clarification_answer",
                f"conversation:clarification_selected:{selected}",
            ]
        else:
            # Interpret the actual turn before deciding whether it is independent
            # or a controlled field-level modifier.
            current_interpretation = interpret_query(
                raw_text=user_text.strip(),
                dataframe=dataframe,
                pipeline1_report=pipeline1_report,
                classification=classification,
                context=previous_intent_context,
            )
            followup = resolve_followup(user_text, context.active_query)
            llm_followup_signals: list[str] = []
            if (
                not followup.is_followup
                and context.active_query is not None
                and self.llm_provider is not None
                and fallback_reason(current_interpretation) is not None
            ):
                llm_followup, llm_followup_signals = interpret_followup_with_llm(
                    user_text,
                    context.active_query.model_dump(mode="json"),
                    classification,
                    pipeline1_report=pipeline1_report,
                    provider=self.llm_provider,
                )
                if llm_followup is not None:
                    followup = self._apply_llm_followup(
                        context.active_query, llm_followup, llm_followup_signals
                    )
            if followup.is_followup:
                assert followup.query is not None
                state = followup.query
                request_text = compile_structured_request(state)
                inherited_signals = list(followup.signals)
            else:
                state = None
                request_text = user_text.strip()
                inherited_signals = ["conversation:new_request", *llm_followup_signals]
                current_interpretation = interpret_query_with_fallback(
                    request_text,
                    dataframe,
                    pipeline1_report=pipeline1_report,
                    classification=classification,
                    context=previous_intent_context,
                    provider=(None if llm_followup_signals else self.llm_provider),
                    deterministic=current_interpretation,
                )

        intent_context = None
        if state is not None and state.intent in {"VISUALIZATION", "INSIGHT"}:
            intent_context = (
                "visualization" if state.intent == "VISUALIZATION" else "insights"
            )
        interpretation = current_interpretation if state is None else None
        if interpretation is None:
            interpretation = interpret_query(
                raw_text=request_text,
                dataframe=dataframe,
                pipeline1_report=pipeline1_report,
                classification=classification,
                context=intent_context,
            )

        intent = state.intent if state is not None else interpretation.query.intent
        if (
            state is None
            and intent == "UNKNOWN"
            and interpretation.query.comparison in {"TOP_N", "BOTTOM_N"}
            and interpretation.query.resolved_measures
            and interpretation.query.resolved_dimensions
        ):
            intent = "INSIGHT"
            inherited_signals.append("conversation:top_bottom_intent_resolved")
        # Explicit complete wording may switch intent; short modifiers never reach here.
        if state is not None and interpretation.query.intent in {"VISUALIZATION", "INSIGHT"}:
            intent = interpretation.query.intent

        if intent == "VISUALIZATION":
            payload = build_visualization_results(
                dataframe=dataframe,
                request_text=request_text,
                pipeline1_report=pipeline1_report,
                dataset_name=dataset_name,
            )
            visualization, insight = payload, None
        elif intent == "INSIGHT":
            payload = build_insight_results(
                dataframe=dataframe,
                question=request_text,
                pipeline1_report=pipeline1_report,
                dataset_name=dataset_name,
                classification=classification,
            )
            visualization, insight = None, payload
        else:
            payload = {
                "status": "NEEDS_CLARIFICATION",
                "message": "Clarify whether you want a visualization or a factual insight.",
                "clarification_options": ["VISUALIZATION", "INSIGHT"],
                "signals": [*interpretation.signals, "conversation:unknown_intent"],
            }
            visualization = insight = None

        status = str(payload.get("status", "INVALID"))
        query_state = self._state_from_interpretation(interpretation, intent)
        query_state = self._apply_effective_result(query_state, payload, intent)
        new_pending = self._pending_from_interpretation(interpretation, query_state)
        options = [str(item) for item in payload.get("clarification_options", [])]
        signals = [
            *inherited_signals,
            *[str(item) for item in payload.get("signals", [])],
        ]
        message = payload.get("message")
        if message is None and status == "READY":
            message = "The deterministic request completed successfully."

        turn_id = str(uuid4())
        turn = ConversationTurn(
            turn_id=turn_id,
            user_text=user_text.strip(),
            intent=intent,
            status=status,
            query=query_state,
            result_metadata=self._result_metadata(payload, intent),
            signals=tuple(signals),
        )
        self.store.update(
            context.dataset_id,
            context.conversation_id,
            active_query=query_state,
            pending_clarification=new_pending,
        )
        self.store.append_turn(
            context.dataset_id, context.conversation_id, turn
        )
        return ConversationResult(
            conversation_id=context.conversation_id,
            turn_id=turn_id,
            intent=intent,
            status=status,
            message=str(message) if message is not None else None,
            visualization=visualization,
            insight=insight,
            clarification_options=options,
            signals=signals,
        )

    @staticmethod
    def _state_from_interpretation(
        interpretation: QueryInterpretation,
        intent: str,
    ) -> ConversationQueryState:
        query = interpretation.query
        return ConversationQueryState(
            intent=intent,
            measure=query.resolved_measures[0] if query.resolved_measures else None,
            dimension=query.resolved_dimensions[0] if query.resolved_dimensions else None,
            measure_term=query.measure_terms[0] if query.measure_terms else None,
            dimension_term=query.dimension_terms[0] if query.dimension_terms else None,
            aggregation=query.aggregation,
            comparison=query.comparison,
            limit=query.limit,
            filters=tuple(query.filters),
            temporal_terms=tuple(query.time_terms),
            requested_chart=query.requested_chart,
            requested_chart_variant=query.requested_chart_variant,
        )

    @staticmethod
    def _apply_effective_result(
        state: ConversationQueryState,
        payload: Mapping[str, Any],
        intent: str,
    ) -> ConversationQueryState:
        if intent == "INSIGHT" and isinstance(payload.get("result"), Mapping):
            result = payload["result"]
            return state.model_copy(update={
                "measure": result.get("measure_column") or state.measure,
                "dimension": result.get("dimension_column") or state.dimension,
                "aggregation": result.get("aggregation", state.aggregation),
                "comparison": result.get("comparison", state.comparison),
            })
        visualizations = payload.get("visualizations")
        if intent == "VISUALIZATION" and isinstance(visualizations, list) and visualizations:
            item = visualizations[0]
            aggregation = item.get("chart_data", {}).get("aggregation")
            update: dict[str, Any] = {
                "requested_chart": item.get("chart_type"),
                "requested_chart_variant": item.get("presentation_variant"),
            }
            if isinstance(aggregation, str):
                update["aggregation"] = aggregation.upper()
            return state.model_copy(update=update)
        return state

    @staticmethod
    def _pending_from_interpretation(
        interpretation: QueryInterpretation,
        state: ConversationQueryState,
    ) -> PendingClarification | None:
        for decision in interpretation.ambiguity_decisions:
            if decision.status != "NEEDS_CLARIFICATION" or not decision.choices:
                continue
            slots = {
                "MEASURE": "measure", "RELATIONSHIP": "measure",
                "DIMENSION": "dimension", "TIME": "time", "FILTER": "filter",
            }
            return PendingClarification(
                term=decision.term,
                slot=slots[decision.kind],
                choices=tuple(decision.choices),
                originating_query=state,
            )
        return None

    @staticmethod
    def _select_clarification(
        user_text: str, pending: PendingClarification
    ) -> str | None:
        answer = normalize_text(user_text)
        for choice in pending.choices:
            if answer == normalize_text(choice):
                return choice
        return None

    @staticmethod
    def _apply_clarification(
        pending: PendingClarification, selected: str
    ) -> ConversationQueryState:
        update: dict[str, Any]
        if pending.slot == "measure":
            update = {"measure": selected, "measure_term": selected}
        elif pending.slot == "dimension":
            update = {"dimension": selected, "dimension_term": selected}
        else:
            update = {}
        return pending.originating_query.model_copy(update=update)

    def _record_controlled_clarification(
        self,
        context: ConversationContext,
        user_text: str,
        pending: PendingClarification,
    ) -> ConversationResult:
        turn_id = str(uuid4())
        message = f"Choose one of the available options for '{pending.term}'."
        signals = ["conversation:clarification_answer_not_recognized"]
        turn = ConversationTurn(
            turn_id=turn_id,
            user_text=user_text.strip(),
            intent=pending.originating_query.intent,
            status="NEEDS_CLARIFICATION",
            query=pending.originating_query,
            signals=tuple(signals),
        )
        self.store.append_turn(context.dataset_id, context.conversation_id, turn)
        return ConversationResult(
            conversation_id=context.conversation_id,
            turn_id=turn_id,
            intent=pending.originating_query.intent,
            status="NEEDS_CLARIFICATION",
            message=message,
            clarification_options=list(pending.choices),
            signals=signals,
        )

    @staticmethod
    def _result_metadata(payload: Mapping[str, Any], intent: str) -> dict[str, Any]:
        metadata: dict[str, Any] = {"status": payload.get("status")}
        if intent == "VISUALIZATION":
            metadata["visualization_count"] = payload.get("visualization_count", 0)
            visualizations = payload.get("visualizations")
            if isinstance(visualizations, list) and visualizations:
                metadata["chart_type"] = visualizations[0].get("chart_type")
        elif isinstance(payload.get("result"), Mapping):
            metadata["query_type"] = payload["result"].get("query_type")
        return metadata

    @staticmethod
    def _apply_llm_followup(
        previous: ConversationQueryState,
        llm: LLMFollowupInterpretation,
        signals: list[str],
    ) -> FollowupResolution:
        update: dict[str, Any] = {}
        if llm.modification == "REPLACE_MEASURE":
            update = {"measure": None, "measure_term": llm.replace_measure_term}
        elif llm.modification == "REPLACE_DIMENSION":
            update = {"dimension": None, "dimension_term": llm.replace_dimension_term}
        elif llm.modification == "SET_AGGREGATION":
            update = {"aggregation": llm.aggregation}
        elif llm.modification == "SET_COMPARISON":
            update = {"comparison": llm.comparison}
        elif llm.modification == "SET_CHART":
            update = {"requested_chart": llm.requested_chart}
        elif llm.modification == "SET_LIMIT":
            update = {"limit": llm.limit}
        elif llm.modification == "ADD_FILTER" and llm.filter is not None:
            update = {"filters": (*previous.filters, QueryFilter(
                raw_text=(
                    f"{llm.filter.column_term} {llm.filter.operator} "
                    f"{llm.filter.value_term}"
                ),
                column_term=llm.filter.column_term,
                operator=llm.filter.operator,
                value_term=llm.filter.value_term,
            ))}
        elif llm.modification == "SET_TEMPORAL":
            update = {"temporal_terms": tuple(llm.temporal_terms)}
        return FollowupResolution(
            is_followup=True,
            modification=f"llm:{llm.modification.casefold()}",
            query=previous.model_copy(update=update),
            signals=tuple([*signals, "conversation:llm_structured_merge"]),
        )


__all__ = ["ConversationOrchestrator"]
