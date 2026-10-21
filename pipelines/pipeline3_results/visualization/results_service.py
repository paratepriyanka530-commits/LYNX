"""Results Service Composition Layer -- Pipeline 3, Stage 10.

Stage 10 of Pipeline 3: the application-facing composition layer for tabular
visualization intelligence.

Combines:
- Stage 6: Automatic visualization orchestration & contract
- Shared NLP: interpretation, evidence-aware resolution, and ambiguity handling
- Stage 7: Visualization compatibility adapter & request validator
- Stage 8: Requested visualization resolution
- Stage 9: Deterministic factual visualization explanations & XAI evidence

Execution Modes:
- Mode A (Automatic): When request_text is None or empty. Calls Stage 6 to
  generate candidate visualizations, enriches each with Stage 9 factual
  explanations, and returns a unified automatic payload.
- Mode B (Requested): When request_text is provided. Runs shared NLP, adapts a
  resolved interpretation to the existing request contract, and validates it.
  If VALID, calls Stage 8 resolver and enriches the resolved chart with Stage 9
  explanation. If clarification or invalidity is detected, returns structured
  signals/options/errors without chart generation.

Invariants:
- Preserves Stage 6 candidate ranking and scores for automatic mode.
- Preserves score = None for user-requested mode (never generates scores).
- Preserves Stage 4/8 chart_data exactly.
- Enforces strict JSON safety with native Python primitives.
- Deterministic: identical inputs produce identical outputs with zero randomness.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

from intelligence.llm.provider import LLMProvider
from pipelines.pipeline3_results.nlp.schemas import QueryInterpretation
from pipelines.pipeline3_results.planning.schemas import AnalyticalPlan

from pipelines.pipeline3_results.visualization.column_classifier import (
    classify_dataframe,
)
from pipelines.pipeline3_results.visualization.explanation import (
    build_visualization_explanation,
)
from pipelines.pipeline3_results.visualization.feasibility import (
    analyze_visualization_feasibility,
)
from pipelines.pipeline3_results.visualization.integration import (
    _to_json_safe,
    build_visualization_payload,
)
# Kept as a module-level compatibility symbol for callers/tests that patch the
# historical resolver. Successful requests no longer invoke it.
from pipelines.pipeline3_results.visualization.request_resolver import (
    resolve_visualization_request,
)
from pipelines.pipeline3_results.visualization.request_validator import (
    validate_visualization_request,
)


def _requested_chart_family(interpretation: QueryInterpretation | None):
    if interpretation is None or interpretation.query.requested_chart is None:
        return None
    from pipelines.pipeline3_results.visualization.intelligence.schemas import ChartFamily
    raw = interpretation.query.requested_chart_variant or interpretation.query.requested_chart
    return ChartFamily("BAR" if raw == "COUNT_BAR" else raw)


def _enrich_with_explanation(
    item: Mapping[str, Any],
    dataset_signals: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Attach Stage 9 explanation sub-dict to a visualization item safely."""
    enriched = dict(item)
    try:
        explanation_res = build_visualization_explanation(
            visualization=item,
            dataset_signals=dataset_signals,
        )
        enriched["explanation"] = _to_json_safe(
            explanation_res.get("explanation", {})
        )
    except Exception:
        enriched["explanation"] = {
            "why_this_chart": "Visualization rationale is unavailable.",
            "what_it_shows": "No displayable chart data available.",
            "aggregation": "Aggregation attribution is unavailable.",
            "data_coverage": "No displayable chart data available.",
            "evidence": [],
        }
    return _to_json_safe(enriched)


def _enrich_with_xai(
    item: Mapping[str, Any],
    *,
    visual,
    verified_result,
    provider: LLMProvider | None,
    dataset_signals: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Attach Phase 9 XAI additively while retaining the legacy explanation."""
    from pipelines.pipeline3_results.visualization.xai.service import (
        VisualizationXAIService,
    )

    enriched = _enrich_with_explanation(item, dataset_signals)
    try:
        xai = VisualizationXAIService().build(
            verified_result, visual.plan, visual.chart_spec, provider=provider,
        )
        xai_payload = xai.model_dump(mode="json")
        enriched["xai"] = xai_payload
        enriched["signals"] = [
            *[str(signal) for signal in enriched.get("signals", [])],
            (
                "xai:deterministic_fallback"
                if xai.fallback_used else "xai:validated_model_wording"
            ),
        ]
        legacy = dict(enriched.get("explanation", {}))
        legacy.update({
            "why_this_chart": xai.why_this_chart,
            "what_it_shows": xai.what_it_shows,
            "limitations": list(xai.limitations),
        })
        enriched["explanation"] = legacy
    except Exception:
        # A wording sidecar must never make an otherwise valid chart fail.
        pass
    return _to_json_safe(enriched)


def build_visualization_results(
    dataframe: pd.DataFrame,
    request_text: str | None = None,
    pipeline1_report: Mapping[str, Any] | None = None,
    dataset_name: str | None = None,
    max_visualizations: int = 5,
    llm_provider: LLMProvider | None = None,
    xai_llm_provider: LLMProvider | None = None,
    interpretation: QueryInterpretation | None = None,
    analytical_plan: AnalyticalPlan | None = None,
) -> dict[str, Any]:
    """Build unified Pipeline 3 visualization results.

    Parameters
    ----------
    dataframe:
        The cleaned / prepared pandas DataFrame to visualize.
    request_text:
        Optional natural-language visualization query (e.g., "Median price by category").
        If None or empty/whitespace-only, automatic mode is executed.
    pipeline1_report:
        Optional Pipeline 1 cleaning / profiling report mapping.
    dataset_name:
        Optional human-readable dataset name identifier.
    max_visualizations:
        Maximum number of visualizations to return for automatic mode (default: 5).
        Must be a positive integer (> 0).
    xai_llm_provider:
        Optional existing-provider implementation used only to word verified XAI
        facts. If omitted, deterministic wording is returned.

    Returns
    -------
    dict[str, Any]
        Unified JSON-safe visualization response contract.

    Raises
    ------
    TypeError:
        If dataframe is not a pandas DataFrame, or request_text / dataset_name /
        pipeline1_report have invalid types.
    ValueError:
        If max_visualizations is not a positive integer (> 0) or is a boolean.
    """
    # -----------------------------------------------------------------------
    # 1. Input Validation
    # -----------------------------------------------------------------------
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            f"dataframe must be a pandas DataFrame, got {type(dataframe).__name__}"
        )

    if request_text is not None and not isinstance(request_text, str):
        raise TypeError(
            f"request_text must be a string or None, got {type(request_text).__name__}"
        )

    if pipeline1_report is not None and not isinstance(pipeline1_report, Mapping):
        raise TypeError(
            f"pipeline1_report must be a Mapping or None, got {type(pipeline1_report).__name__}"
        )

    if dataset_name is not None and (not isinstance(dataset_name, str) or isinstance(dataset_name, bool)):
        raise TypeError(
            f"dataset_name must be a string or None, got {type(dataset_name).__name__}"
        )

    if isinstance(max_visualizations, bool) or not isinstance(max_visualizations, int) or max_visualizations <= 0:
        raise ValueError(
            f"max_visualizations must be a positive integer (> 0), got {max_visualizations!r}"
        )
    if analytical_plan is not None and not isinstance(analytical_plan, AnalyticalPlan):
        raise TypeError("analytical_plan must be an AnalyticalPlan or None")

    clean_dataset_name = str(dataset_name) if dataset_name is not None else None

    # -----------------------------------------------------------------------
    # 2. Execution Routing: Mode A (Automatic) vs Mode B (Requested)
    # -----------------------------------------------------------------------
    is_requested = analytical_plan is not None or (
        request_text is not None and bool(request_text.strip())
    )

    if not is_requested:
        # -------------------------------------------------------------------
        # MODE A: Automatic Visualization
        # -------------------------------------------------------------------
        from pipelines.pipeline3_results.visualization.intelligence.compatibility import to_legacy_item
        from pipelines.pipeline3_results.visualization.intelligence.service import VisualizationIntelligenceService
        # Stage 6 remains a public-contract compatibility source for historical
        # suppression and rank identifiers.  Its chart data is never consumed.
        legacy_contract = build_visualization_payload(
            dataframe=dataframe, pipeline1_report=pipeline1_report,
            dataset_name=clean_dataset_name, max_visualizations=max_visualizations,
        )
        if legacy_contract.get("status") != "READY":
            return _to_json_safe({
                "mode": "automatic",
                "status": "NO_MEANINGFUL_VISUALIZATION",
                "dataset_name": clean_dataset_name,
                "visualization_count": 0,
                "visualizations": [],
                "signals": [
                    *[str(s) for s in legacy_contract.get("signals", [])],
                    "visualization_intelligence:compatibility_suppression",
                ],
            })
        intelligence_service = VisualizationIntelligenceService()
        dashboard = intelligence_service.build_dashboard(
            clean_dataset_name or "dataset", dataframe,
            pipeline1_report=pipeline1_report,
            max_visualizations=max_visualizations,
        )
        top_signals = ["visualization_intelligence:phase7", "automatic:shared_executor"]
        legacy_ranks = [int(item["rank"]) for item in legacy_contract.get("visualizations", [])]
        enriched_visualizations = []
        for index, visual in enumerate(dashboard.visualizations, 1):
            rank = legacy_ranks[index - 1] if index <= len(legacy_ranks) else index
            legacy_item = to_legacy_item(visual, rank=rank)
            verified = intelligence_service.get_verified_result(visual.verified_result_ref)
            enriched_visualizations.append(
                _enrich_with_xai(
                    legacy_item, visual=visual, verified_result=verified,
                    provider=xai_llm_provider, dataset_signals=top_signals,
                ) if verified is not None else _enrich_with_explanation(
                    legacy_item, dataset_signals=top_signals,
                )
            )

        result: dict[str, Any] = {
            "mode": "automatic",
            "status": "READY" if enriched_visualizations else "NO_MEANINGFUL_VISUALIZATION",
            "dataset_name": clean_dataset_name,
            "visualization_count": len(enriched_visualizations),
            "visualizations": enriched_visualizations,
            "kpis": [item.model_dump(mode="json") for item in dashboard.kpis],
            "dashboard": dashboard.model_dump(mode="json"),
            "signals": top_signals,
        }
        return _to_json_safe(result)

    # -----------------------------------------------------------------------
    # MODE B: User-Requested Visualization
    # -----------------------------------------------------------------------
    clean_request_text = request_text.strip() if request_text is not None else ""

    # Guard: empty dataframe for requested mode
    if len(dataframe) == 0:
        return _to_json_safe({
            "mode": "requested",
            "request_text": clean_request_text,
            "status": "NO_MEANINGFUL_VISUALIZATION",
            "dataset_name": clean_dataset_name,
            "visualization_count": 0,
            "visualizations": [],
            "signals": ["dataframe has zero rows -- no chart data can be produced"],
        })

    # Step 1 & 2: Classify and Analyze Feasibility
    classification = classify_dataframe(
        dataframe=dataframe,
        pipeline1_report=pipeline1_report,
        dataset_name=clean_dataset_name,
    )
    feasibility = analyze_visualization_feasibility(
        dataframe=dataframe,
        classification=classification,
    )

    # Step 3: Shared NLP interpretation and visualization adaptation.  The
    # legacy request_parser remains available, but is no longer primary here.
    # Imports are intentionally local: the shared resolver reuses the existing
    # visualization evidence extractor, while visualization.__init__ exports
    # this service.  Local composition avoids an eager package-import cycle.
    from pipelines.pipeline3_results.nlp.semantic_interpreter import (
        interpret_query_pretrained_first,
    )
    from pipelines.pipeline3_results.visualization.nlp_adapter import (
        adapt_query_interpretation,
    )

    active_plan = analytical_plan
    active_interpretation = interpretation
    if active_plan is None:
        active_interpretation = active_interpretation or interpret_query_pretrained_first(
            clean_request_text,
            dataframe,
            pipeline1_report=pipeline1_report,
            classification=classification,
            context="visualization",
            provider=llm_provider,
        )
        if (
            active_interpretation.query.intent == "VISUALIZATION"
            and active_interpretation.analytical_plan is not None
        ):
            active_plan = AnalyticalPlan.model_validate(
                active_interpretation.analytical_plan
            )

    if active_plan is not None:
        from pipelines.pipeline3_results.execution.pandas_executor import (
            PandasAnalyticalExecutor,
        )
        from pipelines.pipeline3_results.grounding.dataset_knowledge import (
            build_dataset_knowledge,
        )
        from pipelines.pipeline3_results.visualization.intelligence.compatibility import to_legacy_item
        from pipelines.pipeline3_results.visualization.intelligence.service import build_requested_visualization

        knowledge = build_dataset_knowledge(
            pipeline1_report,
            dataset_id=clean_dataset_name,
            dataframe_schema={str(name): str(dtype) for name, dtype in dataframe.dtypes.items()},
        )
        verified = PandasAnalyticalExecutor().execute(
            active_plan, dataframe, knowledge
        )
        shared_signals = [
            f"analytical_plan:{active_plan.plan_id}",
            f"shared_executor:{verified.execution_metadata.backend}",
            f"verified_result:{verified.result_id}",
            f"verification:{verified.verification.status}",
            "explicit user request",
        ]
        if verified.status != "READY":
            return _to_json_safe({
                "mode": "requested",
                "request_text": clean_request_text,
                "status": (
                    "NO_MEANINGFUL_VISUALIZATION"
                    if verified.status == "EMPTY" else "NEEDS_CLARIFICATION"
                ),
                "failure_code": (
                    verified.errors[0].code if verified.errors else None
                ),
                "dataset_name": clean_dataset_name,
                "visualization_count": 0,
                "visualizations": [],
                "message": verified.errors[0].message if verified.errors else None,
                "signals": [*shared_signals, "visualization_service:shared_execution_stopped"],
            })
        assessment, visual = build_requested_visualization(
            active_plan, verified, knowledge,
            requested_family=_requested_chart_family(active_interpretation),
        )
        if visual is None:
            return _to_json_safe({
                "mode": "requested",
                "request_text": clean_request_text,
                "status": "INVALID" if assessment.status == "INVALID" else "NO_MEANINGFUL_VISUALIZATION",
                "failure_code": assessment.reason_codes[0] if assessment.reason_codes else None,
                "dataset_name": clean_dataset_name,
                "visualization_count": 0,
                "visualizations": [],
                "message": assessment.message,
                "warnings": [assessment.message] if assessment.status == "NOT_RECOMMENDED" else [],
                "signals": [*shared_signals, "visualization_intelligence:suitability_stopped"],
            })
        item = to_legacy_item(visual, rank=1)
        enriched = _enrich_with_xai(
            item, visual=visual, verified_result=verified,
            provider=xai_llm_provider, dataset_signals=shared_signals,
        )
        return _to_json_safe({
            "mode": "requested",
            "request_text": clean_request_text,
            "status": "READY",
            "dataset_name": clean_dataset_name,
            "visualization_count": 1,
            "visualizations": [enriched],
            "signals": [*shared_signals, "visualization_intelligence:shared_result_consumed"],
        })

    assert active_interpretation is not None
    adaptation = adapt_query_interpretation(active_interpretation, classification)

    if adaptation.status == "NEEDS_CLARIFICATION":
        return _to_json_safe({
            "mode": "requested",
            "request_text": clean_request_text,
            "status": "NEEDS_CLARIFICATION",
            "failure_code": adaptation.failure_code,
            "dataset_name": clean_dataset_name,
            "visualization_count": 0,
            "visualizations": [],
            "message": adaptation.message,
            "ambiguous_terms": [str(term) for term in adaptation.ambiguous_terms],
            "clarification_options": [
                str(option) for option in adaptation.clarification_options
            ],
            "signals": [str(signal) for signal in adaptation.signals],
        })

    if adaptation.status == "INVALID":
        return _to_json_safe({
            "mode": "requested",
            "request_text": clean_request_text,
            "status": "INVALID",
            "failure_code": adaptation.failure_code,
            "dataset_name": clean_dataset_name,
            "visualization_count": 0,
            "visualizations": [],
            "message": adaptation.message,
            "errors": [str(error) for error in adaptation.errors],
            "signals": [str(signal) for signal in adaptation.signals],
        })

    assert adaptation.request is not None

    # Step 4: Validate Request
    validated = validate_visualization_request(
        dataframe=dataframe,
        classification=classification,
        request=adaptation.request,
        feasibility=feasibility,
    )

    # Step 5: Check Validation Status
    if validated.status == "NEEDS_CLARIFICATION":
        clarification_res: dict[str, Any] = {
            "mode": "requested",
            "request_text": clean_request_text,
            "status": "NEEDS_CLARIFICATION",
            "failure_code": (
                "AMBIGUOUS_CONCEPT"
                if validated.clarification_options
                else "INVALID_COMBINATION"
            ),
            "dataset_name": clean_dataset_name,
            "visualization_count": 0,
            "visualizations": [],
            "clarification_options": [str(opt) for opt in validated.clarification_options],
            "signals": [str(s) for s in validated.signals],
        }
        return _to_json_safe(clarification_res)

    if validated.status == "INVALID":
        invalid_res: dict[str, Any] = {
            "mode": "requested",
            "request_text": clean_request_text,
            "status": "INVALID",
            "failure_code": "INVALID_COMBINATION",
            "dataset_name": clean_dataset_name,
            "visualization_count": 0,
            "visualizations": [],
            "message": (
                str(validated.errors[0])
                if validated.errors
                else "The requested chart and fields are not compatible."
            ),
            "errors": [str(e) for e in validated.errors],
            "signals": [str(s) for s in validated.signals],
        }
        return _to_json_safe(invalid_res)

    # A valid legacy request is now only an input compatibility contract.  It
    # is converted to a typed plan and executed by the shared executor; the
    # legacy request resolver no longer owns analytical truth.
    from pipelines.pipeline3_results.execution.pandas_executor import PandasAnalyticalExecutor
    from pipelines.pipeline3_results.grounding.dataset_knowledge import build_dataset_knowledge
    from pipelines.pipeline3_results.visualization.intelligence.compatibility import to_legacy_item
    from pipelines.pipeline3_results.visualization.intelligence.schemas import ChartFamily
    from pipelines.pipeline3_results.visualization.intelligence.service import (
        build_requested_visualization, plan_from_validated_request,
    )
    compatible_plan = plan_from_validated_request(validated)
    knowledge = build_dataset_knowledge(
        pipeline1_report, dataset_id=clean_dataset_name,
        dataframe_schema={str(name): str(dtype) for name, dtype in dataframe.dtypes.items()},
    )
    verified = PandasAnalyticalExecutor().execute(compatible_plan, dataframe, knowledge)
    requested_raw = validated.presentation_variant or validated.chart_type
    requested_family = ChartFamily("BAR" if requested_raw == "COUNT_BAR" else requested_raw)
    shared_signals = [
        *[str(s) for s in validated.signals],
        f"analytical_plan:{compatible_plan.plan_id}",
        f"verified_result:{verified.result_id}",
        "visualization_intelligence:legacy_request_adapted",
        "explicit user request",
    ]
    if verified.status == "READY":
        assessment, visual = build_requested_visualization(
            compatible_plan, verified, knowledge, requested_family,
        )
        if visual is not None:
            item = _enrich_with_xai(
                to_legacy_item(visual, rank=1), visual=visual,
                verified_result=verified, provider=xai_llm_provider,
                dataset_signals=shared_signals,
            )
            return _to_json_safe({
                "mode": "requested", "request_text": clean_request_text,
                "status": "READY", "dataset_name": clean_dataset_name,
                "visualization_count": 1, "visualizations": [item],
                "signals": [*shared_signals, "visualization_intelligence:shared_result_consumed"],
            })
        return _to_json_safe({
            "mode": "requested", "request_text": clean_request_text,
            "status": "INVALID" if assessment.status == "INVALID" else "NO_MEANINGFUL_VISUALIZATION",
            "failure_code": assessment.reason_codes[0] if assessment.reason_codes else None,
            "dataset_name": clean_dataset_name, "visualization_count": 0,
            "visualizations": [], "message": assessment.message,
            "warnings": [assessment.message] if assessment.status != "INVALID" else [],
            "signals": [*shared_signals, "visualization_intelligence:suitability_stopped"],
        })
    return _to_json_safe({
        "mode": "requested", "request_text": clean_request_text,
        "status": "NO_MEANINGFUL_VISUALIZATION" if verified.status == "EMPTY" else "INVALID",
        "failure_code": verified.errors[0].code if verified.errors else None,
        "dataset_name": clean_dataset_name, "visualization_count": 0,
        "visualizations": [],
        "message": verified.errors[0].message if verified.errors else "The analytical result could not be visualized.",
        "signals": [*shared_signals, "visualization_intelligence:execution_stopped"],
    })
