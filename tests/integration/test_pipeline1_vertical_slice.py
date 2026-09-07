import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pipelines.pipeline1_cleaning.decisions.decision_schema import CleaningDecision
from pipelines.pipeline1_cleaning.pipeline import run_pipeline1
from pipelines.pipeline1_cleaning.semantics.batched_schema import (
    BatchedColumnSemantic,
    BatchedDatasetContext,
    SemanticProfile,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET = PROJECT_ROOT / "tests" / "datasets" / "valid" / "dataset_13_generic_business_data_raw.csv"


class Pipeline1VerticalSliceIntegrationTests(unittest.TestCase):
    def test_existing_dataset_produces_identical_no_op_output_and_report(self):
        raw_before = DATASET.read_bytes()
        original = pd.read_csv(DATASET)
        with tempfile.TemporaryDirectory() as temp_dir:
            def mock_semantics(profile, evidence):
                return SemanticProfile(
                    dataset_context=BatchedDatasetContext(
                        row_entity=None, row_entity_confidence=0.0,
                    ),
                    columns=[
                        BatchedColumnSemantic(
                            column_name=name, semantic_type="unknown",
                            value_function="unknown",
                            numeric_meaning="unknown",
                            concept_description="Mock semantic result.",
                            semantic_role="unknown", confidence=0.0,
                            reasoning_summary="Live semantics are excluded from this mechanical test.",
                        )
                        for name in profile["column_names"]
                    ],
                    telemetry={"total_llm_calls": 0},
                )

            def mock_decisions(issues, semantics):
                decisions = [
                    CleaningDecision(
                        decision_id=f"decision-{issue.issue_id}",
                        source_issue_ids=[issue.issue_id],
                        target_columns=issue.target_columns,
                        decision_type="no_action", proposed_action="no_op",
                        confidence=1.0, rationale="Mock conservative decision.",
                        evidence_summary=issue.evidence, semantic_basis={},
                        risk_level="low", requires_review=False,
                        decision_source="deterministic_tier1",
                    )
                    for issue in issues
                ]
                return decisions, {"total_decisions": len(decisions), "llm_decision_calls": 0}, []

            report = run_pipeline1(
                DATASET, temp_dir, semantic_analyzer=mock_semantics,
                decision_engine=mock_decisions,
            )
            cleaned_path = Path(temp_dir) / "cleaned_dataset.csv"
            report_path = Path(temp_dir) / "cleaning_report.json"
            cleaned = pd.read_csv(cleaned_path)
            saved_report = json.loads(report_path.read_text(encoding="utf-8"))

            pd.testing.assert_frame_equal(cleaned, original)
            self.assertEqual(DATASET.read_bytes(), raw_before)
            self.assertTrue(report.validation.dataframe_equal)
            self.assertTrue(report.validation.passed)
            self.assertTrue(all(
                decision["proposed_action"] == "no_op"
                for decision in saved_report["decisions"]
            ))
            self.assertGreater(len(saved_report["issues"]), 0)
            self.assertEqual(
                len(saved_report["semantic_profile"]["columns"]),
                len(original.columns),
            )
            self.assertEqual(
                saved_report["quality_summary"]["total_issues_detected"],
                len(saved_report["issues"]),
            )
            self.assertTrue(cleaned_path.is_file())
            self.assertTrue(report_path.is_file())

    def test_pipeline_executes_only_approved_trim_decisions(self):
        original = pd.DataFrame({"name": [" A ", "B"], "amount": [1, 2]})
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "raw.csv"
            output_path = Path(temp_dir) / "output"
            original.to_csv(input_path, index=False)

            def mock_semantics(profile, evidence):
                return SemanticProfile(
                    dataset_context=BatchedDatasetContext(
                        row_entity="record", row_entity_confidence=1.0,
                    ),
                    columns=[BatchedColumnSemantic(
                        column_name=name, semantic_type="unknown",
                        value_function="unknown", numeric_meaning="unknown",
                        concept_description="Test values.",
                        semantic_role="unknown", confidence=0.8,
                        reasoning_summary="Test semantic context.",
                    ) for name in profile["column_names"]],
                )

            def mock_decisions(issues, semantics):
                decisions = []
                for detected in issues:
                    is_whitespace = detected.issue_type == "whitespace_issue"
                    decisions.append(CleaningDecision(
                        decision_id=f"decision-{detected.issue_id}",
                        source_issue_ids=[detected.issue_id],
                        target_columns=detected.target_columns if is_whitespace else [],
                        decision_type="action" if is_whitespace else "no_action",
                        proposed_action="trim_whitespace" if is_whitespace else "no_op",
                        confidence=1.0, rationale="Test routing.", risk_level="low",
                        requires_review=False,
                        decision_source="deterministic_tier1",
                    ))
                return decisions, {"total_decisions": len(decisions)}, []

            report = run_pipeline1(
                input_path, output_path, semantic_analyzer=mock_semantics,
                decision_engine=mock_decisions,
            )
            cleaned = pd.read_csv(output_path / "cleaned_dataset.csv")

            self.assertEqual(cleaned["name"].tolist(), ["A", "B"])
            self.assertEqual(original["name"].tolist(), [" A ", "B"])
            self.assertEqual([item.action_id for item in report.instructions], ["trim_whitespace"])
            self.assertEqual(report.executions[0].changed_cell_count, 1)
            self.assertTrue(report.validation.targeted_whitespace_improved)
            self.assertTrue(report.validation.unrelated_columns_equal)

    def test_trim_precedes_exact_categorical_normalization(self):
        original = pd.DataFrame({
            "segment": ([" sales ", "sales", "Sales", "sales team"] * 6),
            "amount": list(range(24)),
        })
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "raw.csv"
            output_path = Path(temp_dir) / "output"
            original.to_csv(input_path, index=False)

            def mock_semantics(profile, evidence):
                return SemanticProfile(
                    dataset_context=BatchedDatasetContext(
                        row_entity="record", row_entity_confidence=1.0,
                    ),
                    columns=[BatchedColumnSemantic(
                        column_name=name, semantic_type="categorical",
                        value_function="classification",
                        numeric_meaning="not_applicable",
                        concept_description="Explicit test labels.",
                        semantic_role="unknown", confidence=1.0,
                        reasoning_summary="Test semantic context.",
                    ) for name in profile["column_names"]],
                )

            def mock_decisions(issues, semantics):
                decisions = []
                for detected in issues:
                    if detected.issue_type == "whitespace_issue":
                        action, parameters = "trim_whitespace", {}
                    elif detected.issue_type == "case_variant_candidate":
                        action, parameters = (
                            "normalize_categorical_labels", {"mapping": {"sales": "Sales"}}
                        )
                    else:
                        action, parameters = "no_op", {}
                    decisions.append(CleaningDecision(
                        decision_id=f"decision-{detected.issue_id}",
                        source_issue_ids=[detected.issue_id],
                        target_columns=detected.target_columns if action != "no_op" else [],
                        decision_type="action" if action != "no_op" else "no_action",
                        proposed_action=action, parameters=parameters,
                        confidence=1.0, rationale="Test routing.", risk_level="low",
                        requires_review=False, decision_source="deterministic_tier1",
                    ))
                return decisions, {"total_decisions": len(decisions)}, []

            report = run_pipeline1(
                input_path, output_path, semantic_analyzer=mock_semantics,
                decision_engine=mock_decisions,
            )
            cleaned = pd.read_csv(output_path / "cleaned_dataset.csv")

            self.assertEqual(
                [item.action_id for item in report.instructions[:2]],
                ["trim_whitespace", "normalize_categorical_labels"],
            )
            self.assertEqual(set(cleaned["segment"]), {"Sales", "sales team"})
            self.assertTrue(report.validation.targeted_case_variants_improved)
            self.assertTrue(report.validation.only_approved_value_changes)
            self.assertTrue(report.validation.no_unexpected_categories)
            self.assertFalse(report.validation.cardinality_equal)


if __name__ == "__main__":
    unittest.main()
