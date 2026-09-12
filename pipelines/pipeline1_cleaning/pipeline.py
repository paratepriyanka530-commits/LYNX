"""End-to-end Pipeline 1 mechanical vertical slice."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from pipelines.pipeline1_cleaning.techniques.date_parsing import OUTPUT_FORMATS, assess_dates
from pipelines.pipeline1_cleaning.techniques.numeric_coercion import assess_numeric
from pipelines.pipeline1_cleaning.techniques.missing_values import (
    assess_median_semantic_eligibility,
    missingness_assessment,
)

from pipelines.pipeline1_cleaning.decisions.decision_schema import (
    CleaningDecision,
    Pipeline1Report,
)
from pipelines.pipeline1_cleaning.decisions.cleaning_decision_engine import (
    decide_cleaning,
)
from pipelines.pipeline1_cleaning.ingestion.dataset_loader import load_dataset
from pipelines.pipeline1_cleaning.ingestion.file_validator import validate_file
from pipelines.pipeline1_cleaning.orchestration.action_validator import (
    compile_instruction,
)
from pipelines.pipeline1_cleaning.orchestration.action_registry import ACTION_REGISTRY, get_action
from pipelines.pipeline1_cleaning.orchestration.data_cleaner_performer import (
    execute_instruction,
)
from pipelines.pipeline1_cleaning.output.clean_dataset_exporter import (
    export_pipeline_outputs,
)
from pipelines.pipeline1_cleaning.profiling.dataset_profiler import profile_dataset
from pipelines.pipeline1_cleaning.quality.issue_detector import detect_issues
from pipelines.pipeline1_cleaning.quality.quality_report import summarize_issues
from pipelines.pipeline1_cleaning.semantics.batched_analyzer import (
    analyze_batched_semantics,
)
from pipelines.pipeline1_cleaning.semantics.contradiction_guard import (
    apply_semantic_contradiction_guard,
)
from pipelines.pipeline1_cleaning.semantics.evidence_builder import (
    build_dataset_evidence,
)
from pipelines.pipeline1_cleaning.validation.post_clean_validator import (
    validate_post_clean,
)


PIPELINE_VERSION = "pipeline1_architecture_b_v001"
PIPELINE_FROZEN_FOR_BENCHMARK = True


def _run_default_semantic_analyzer(dataset_profile, dataset_evidence):
    from intelligence.llm.ollama_provider import OllamaProvider
    from intelligence.rag.retriever import RAGRetriever

    retriever = RAGRetriever(
        knowledge_directory=str(Path(__file__).resolve().parents[2] / "knowledge_base")
    )
    retriever.index_knowledge()
    return analyze_batched_semantics(
        dataset_profile=dataset_profile,
        dataset_evidence=dataset_evidence,
        retriever=retriever,
        llm_provider=OllamaProvider(model="qwen2.5:3b"),
    )


def _run_default_decision_engine(issues, semantic_profile, dataframe=None):
    from intelligence.llm.ollama_provider import OllamaProvider

    return decide_cleaning(
        issues=issues,
        semantic_profile=semantic_profile,
        llm_provider=OllamaProvider(model="qwen2.5:3b"),
        dataframe=dataframe,
    )


def run_pipeline1(
    input_path: str | Path,
    output_directory: str | Path,
    semantic_analyzer=None,
    decision_engine=None,
) -> Pipeline1Report:
    input_file = Path(input_path).resolve()
    output_dir = Path(output_directory).resolve()
    cleaned_path = output_dir / "cleaned_dataset.csv"
    report_path = output_dir / "cleaning_report.json"
    if input_file in {cleaned_path, report_path}:
        raise ValueError("Output artifacts must not overwrite the raw input dataset.")

    validation = validate_file(str(input_file))
    if not validation.get("valid"):
        raise ValueError(f"Invalid input dataset: {validation.get('reason')}")

    loaded = load_dataset(str(input_file))
    original = loaded["dataframe"]
    before_profile = profile_dataset(original)
    detected_issues = detect_issues(original, before_profile)
    quality_summary = summarize_issues(detected_issues)
    dataset_evidence = build_dataset_evidence(before_profile)
    analyzer = semantic_analyzer or _run_default_semantic_analyzer
    semantic_profile = analyzer(before_profile, dataset_evidence)
    semantic_profile = apply_semantic_contradiction_guard(semantic_profile, original)
    if decision_engine is None:
        issue_decisions, decision_summary, decision_warnings = _run_default_decision_engine(
            detected_issues, semantic_profile, original,
        )
    else:
        issue_decisions, decision_summary, decision_warnings = decision_engine(
            detected_issues, semantic_profile,
        )

    instructions, execution_warnings, skipped = [], [], []
    semantics_by_column = {item.column_name: item.model_dump(mode="json") for item in semantic_profile.columns}
    date_assessments = []
    numeric_assessments = []
    imputation_decisions = []
    median_authorization_assessments = []
    missing_assessments = [
        {"column": str(column), **missingness_assessment(original[column]),
         "semantic_type": semantics_by_column.get(str(column), {}).get("semantic_type"),
         "value_function": semantics_by_column.get(str(column), {}).get("value_function")}
        for column in original.columns if original[column].isna().any()
    ]
    for decision in issue_decisions:
        if decision.proposed_action == "impute_missing":
            imputation_decisions.append(decision)
            continue
        if decision.proposed_action == "coerce_dtype":
            for column in decision.target_columns:
                if column in original.columns:
                    _, stats = assess_numeric(original[column])
                    semantic = semantics_by_column.get(column, {})
                    numeric_assessments.append({
                        "decision_id": decision.decision_id, "column": column,
                        "semantic_type": semantic.get("semantic_type"),
                        "value_function": semantic.get("value_function"),
                        "requested_target_dtype": decision.parameters.get("target_dtype"),
                        "dtype_before": str(original[column].dtype), **stats,
                    })
        if decision.proposed_action == "parse_dates":
            for column in decision.target_columns:
                kind = semantics_by_column.get(column, {}).get("semantic_type")
                if column in original.columns and kind in OUTPUT_FORMATS:
                    _, stats = assess_dates(original[column], OUTPUT_FORMATS[kind])
                    date_assessments.append({"decision_id": decision.decision_id, "column": column, **stats})
        executable = (
            decision.decision_type == "action"
            and decision.proposed_action in {
                "trim_whitespace", "normalize_categorical_labels", "parse_dates", "coerce_dtype",
            }
            and decision.proposed_action in ACTION_REGISTRY
            and not decision.requires_review
            and decision.risk_level == "low"
        )
        if executable:
            try:
                instructions.append(compile_instruction(
                    decision, list(map(str, original.columns)),
                    {str(column): dtype for column, dtype in original.dtypes.items()},
                    dataframe=original, column_semantics=semantics_by_column,
                ))
            except ValueError as exc:
                skipped.append({
                    "decision_id": decision.decision_id,
                    "proposed_action": decision.proposed_action,
                    "reason": str(exc),
                })
        elif decision.decision_type == "action":
            skipped.append({
                "decision_id": decision.decision_id,
                "proposed_action": decision.proposed_action,
                "reason": "Action is not approved and executable in the current MVP.",
            })

    action_priority = {"no_op": 0, "trim_whitespace": 10, "normalize_categorical_labels": 20,
                       "parse_dates": 30, "coerce_dtype": 40, "impute_missing": 50}
    for decision in imputation_decisions:
        original_recommendation = {
            "decision_type": decision.decision_type,
            "proposed_action": decision.proposed_action,
            "parameters": dict(decision.parameters),
            "confidence": decision.confidence,
            "rationale": decision.rationale,
            "risk_level": decision.risk_level,
            "requires_review": decision.requires_review,
            "decision_source": decision.decision_source,
        }
        authorization = {
            "decision_id": decision.decision_id,
            "source_issue_ids": list(decision.source_issue_ids),
            "target_columns": list(decision.target_columns),
            "llm_recommendation": original_recommendation,
            "semantic_eligibility": None,
            "mechanical_eligibility": {
                "status": "not_evaluated",
                "reason_code": "not_evaluated",
                "reason": "Mechanical eligibility was not evaluated.",
            },
            "final_adjudication": {
                "authorized": False,
                "risk_level": decision.risk_level,
                "requires_review": decision.requires_review,
                "reason_code": "existing_policy",
                "reason": "The existing action policy remains in force.",
            },
        }
        is_tier2_median_candidate = (
            decision.decision_source == "llm_tier2"
            and decision.decision_type == "action"
            and decision.proposed_action == "impute_missing"
            and decision.parameters.get("strategy") == "median"
        )
        try:
            # Validate against the values that will reach the final action.
            # Preview only already approved preceding transformations on a copy.
            preview = original.copy(deep=True)
            for preceding in sorted(instructions, key=lambda item: action_priority[item.action_id]):
                if set(preceding.target_columns) & set(decision.target_columns):
                    if preceding.action_id == "impute_missing":
                        raise ValueError("Only one imputation instruction per target column is allowed.")
                    get_action(preceding.action_id)(preview, preceding.target_columns, preceding.parameters)

            adjudicated_decision = decision
            if is_tier2_median_candidate:
                if len(decision.target_columns) != 1 or decision.target_columns[0] not in preview.columns:
                    semantic_result = {
                        "semantic_eligible": "unknown",
                        "reason_code": "semantic_evidence_unavailable",
                        "reason": "Median semantic eligibility requires exactly one existing target column.",
                        "evidence": {"target_columns": list(decision.target_columns)},
                    }
                else:
                    column = decision.target_columns[0]
                    semantic_result = assess_median_semantic_eligibility(
                        semantics_by_column.get(column, {}),
                        preview[column],
                        decision.evidence_summary,
                    )
                authorization["semantic_eligibility"] = semantic_result
                if semantic_result["semantic_eligible"] != "true":
                    authorization["mechanical_eligibility"]["reason"] = (
                        "Mechanical eligibility was not evaluated because semantic eligibility did not authorize it."
                    )
                    authorization["final_adjudication"] = {
                        "authorized": False,
                        "risk_level": decision.risk_level,
                        "requires_review": decision.requires_review,
                        "reason_code": semantic_result["reason_code"],
                        "reason": semantic_result["reason"],
                    }
                    median_authorization_assessments.append(authorization)
                    skipped.append({
                        "decision_id": decision.decision_id,
                        "source_issue_ids": decision.source_issue_ids,
                        "target_columns": decision.target_columns,
                        "proposed_action": decision.proposed_action,
                        "strategy": decision.parameters.get("strategy"),
                        "rejection_stage": "semantic_eligibility",
                        "reason_code": semantic_result["reason_code"],
                        "reason": semantic_result["reason"],
                    })
                    continue
                adjudicated_decision = decision.model_copy(update={
                    "risk_level": "low", "requires_review": False,
                })

            instruction = compile_instruction(
                adjudicated_decision, list(map(str, preview.columns)),
                {str(column): dtype for column, dtype in preview.dtypes.items()},
                dataframe=preview, column_semantics=semantics_by_column,
            )
            instructions.append(instruction)
            if is_tier2_median_candidate:
                authorization["mechanical_eligibility"] = {
                    "status": "passed",
                    "reason_code": "existing_imputation_contract_passed",
                    "reason": "The existing deterministic imputation compiler validation passed completely.",
                }
                authorization["final_adjudication"] = {
                    "authorized": True,
                    "risk_level": "low",
                    "requires_review": False,
                    "reason_code": "semantic_and_mechanical_eligibility_passed",
                    "reason": "Both semantic eligibility and the existing mechanical contract passed.",
                }
                median_authorization_assessments.append(authorization)
                decision_index = next(
                    index for index, item in enumerate(issue_decisions)
                    if item.decision_id == decision.decision_id
                )
                issue_decisions[decision_index] = adjudicated_decision
        except (ValueError, TypeError) as exc:
            if is_tier2_median_candidate:
                authorization["mechanical_eligibility"] = {
                    "status": "failed",
                    "reason_code": "existing_imputation_contract_failed",
                    "reason": str(exc),
                }
                authorization["final_adjudication"] = {
                    "authorized": False,
                    "risk_level": decision.risk_level,
                    "requires_review": decision.requires_review,
                    "reason_code": "mechanical_eligibility_failed",
                    "reason": str(exc),
                }
                median_authorization_assessments.append(authorization)
            skipped.append({
                "decision_id": decision.decision_id, "source_issue_ids": decision.source_issue_ids,
                "target_columns": decision.target_columns, "proposed_action": decision.proposed_action,
                "strategy": decision.parameters.get("strategy"),
                "rejection_stage": "mechanical_eligibility" if is_tier2_median_candidate else "compiler",
                "reason_code": "existing_imputation_contract_failed" if is_tier2_median_candidate else "compiler_rejected",
                "reason": str(exc),
            })

    if not instructions:
        execution_guard = CleaningDecision(
            decision_id="decision-execution-safety-no-op",
            source_issue_ids=[issue.issue_id for issue in detected_issues] or ["no-detected-issues"],
            decision_type="no_action", proposed_action="no_op", confidence=1.0,
            rationale="No approved executable cleaning actions were available.",
            evidence_summary={"execution_fallback": True}, semantic_basis={},
            risk_level="low", requires_review=False,
            decision_source="deterministic_tier1",
        )
        instructions.append(
            compile_instruction(execution_guard, list(map(str, original.columns)))
        )

    instructions.sort(key=lambda item: action_priority[item.action_id])

    cleaned = original.copy(deep=True)
    executions = []
    for instruction in instructions:
        cleaned, execution = execute_instruction(cleaned, instruction)
        executions.append(execution)
        if execution.status == "failed":
            execution_warnings.append(execution.error or execution.message)

    modified_columns = sorted({
        column for instruction in instructions
        if instruction.action_id in {
            "trim_whitespace", "normalize_categorical_labels", "parse_dates", "coerce_dtype", "impute_missing",
        }
        for column in instruction.target_columns
    })
    validation_report, _ = validate_post_clean(
        original=original,
        cleaned=cleaned,
        before_profile=before_profile,
        expect_no_changes=not modified_columns,
        modified_columns=modified_columns,
        executed_instructions=instructions,
    )
    decision_summary["execution_routing"] = {
        "detected_issue_count": len(detected_issues),
        "decision_count": len(issue_decisions),
        "proposed_action_count": sum(
            decision.decision_type == "action" for decision in issue_decisions
        ),
        "compiled_instruction_count": len(instructions),
        "executed_action_count": len(executions),
        "executed_trim_whitespace_count": sum(
            execution.action_id == "trim_whitespace" for execution in executions
        ),
        "executed_normalize_categorical_labels_count": sum(
            execution.action_id == "normalize_categorical_labels"
            for execution in executions
        ),
        "skipped_action_count": len(skipped),
        "skipped_actions": skipped,
        "parse_date_assessments": date_assessments,
        "numeric_coercion_assessments": numeric_assessments,
        "missing_value_assessments": missing_assessments,
        "impute_missing_proposal_count": len(imputation_decisions),
        "median_imputation_authorization_assessments": median_authorization_assessments,
        "executed_impute_missing_count": sum(
            execution.action_id == "impute_missing" and execution.status == "success"
            for execution in executions
        ),
        "executed_coerce_dtype_count": sum(
            execution.action_id == "coerce_dtype" and execution.status == "success"
            for execution in executions
        ),
        "executed_parse_dates_count": sum(
            execution.action_id == "parse_dates" and execution.status == "success"
            for execution in executions
        ),
    }

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    report = Pipeline1Report(
        pipeline_version=PIPELINE_VERSION,
        run_id=run_id,
        status="success" if validation_report.passed and not execution_warnings else "failed",
        original_dataset={
            **validation,
            "source_path": str(input_file),
            "raw_dataset_overwritten": False,
        },
        before_profile=before_profile,
        issues=[issue.model_dump(mode="json") for issue in detected_issues],
        quality_summary=quality_summary,
        semantic_profile=semantic_profile.model_dump(mode="json"),
        decision_summary=decision_summary,
        decisions=issue_decisions,
        instructions=instructions,
        executions=executions,
        validation=validation_report,
        warnings=(
            list(validation_report.warnings)
            + list(semantic_profile.warnings)
            + list(decision_warnings)
            + execution_warnings
        ),
        output_artifacts={},
    )
    export_pipeline_outputs(cleaned, report, output_dir)
    return report
