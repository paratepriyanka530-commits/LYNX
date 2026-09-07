import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pipelines.pipeline1_cleaning.decisions.decision_schema import CleaningDecision
from pipelines.pipeline1_cleaning.orchestration.action_validator import compile_instruction
from pipelines.pipeline1_cleaning.orchestration.data_cleaner_performer import execute_instruction
from pipelines.pipeline1_cleaning.pipeline import run_pipeline1
from pipelines.pipeline1_cleaning.profiling.dataset_profiler import profile_dataset
from pipelines.pipeline1_cleaning.quality.issue_detector import detect_issues
from pipelines.pipeline1_cleaning.semantics.batched_schema import SemanticProfile
from pipelines.pipeline1_cleaning.validation.post_clean_validator import validate_post_clean


def decision(kind="numeric_measurement", function="measurement", parameters=None, **overrides):
    fields = dict(
        decision_id="numeric-decision", source_issue_ids=["numeric-issue"], target_columns=["value"],
        decision_type="action", proposed_action="coerce_dtype", parameters=parameters or {},
        semantic_basis={"value": {"semantic_type": kind, "value_function": function}},
        confidence=0.98, rationale="Approved quantitative values.", risk_level="low",
        requires_review=False, decision_source="deterministic_tier1",
    )
    fields.update(overrides)
    return CleaningDecision(**fields)


def compile_for(frame, proposal=None):
    return compile_instruction(proposal or decision(), list(frame.columns), dict(frame.dtypes), dataframe=frame)


class CoerceDtypeTests(unittest.TestCase):
    def test_integer_float_nulls_lineage_and_raw_preservation(self):
        for values, target, expected in (
            (["12", "15", "19.0", None, pd.NA], "integer", [12, 15, 19]),
            (["12.5", "15", "19.1", None, pd.NA], "float", [12.5, 15.0, 19.1]),
        ):
            with self.subTest(target=target):
                raw = pd.DataFrame({"value": pd.Series(values, dtype=object), "other": range(5)})
                snapshot = raw.copy(deep=True)
                instruction = compile_for(raw)
                cleaned, execution = execute_instruction(raw, instruction)
                self.assertEqual(instruction.parameters, {"target_dtype": target})
                self.assertEqual(cleaned.value.iloc[:3].tolist(), expected)
                self.assertEqual(str(cleaned.value.dtype), "Int64" if target == "integer" else "Float64")
                pd.testing.assert_frame_equal(raw, snapshot)
                pd.testing.assert_series_equal(raw.other, cleaned.other)
                self.assertTrue(raw.isna().equals(cleaned.isna()))
                self.assertEqual(execution.changed_cell_count, 3)
                self.assertEqual(execution.source_issue_ids, ["numeric-issue"])
                self.assertEqual(execution.source_decision_id, "numeric-decision")
                self.assertEqual(execution.status, "success")
                report, _ = validate_post_clean(raw, cleaned, profile_dataset(raw),
                    modified_columns=["value"], executed_instructions=[instruction])
                self.assertTrue(report.passed)
                self.assertFalse(report.physical_dtypes_equal)
                self.assertTrue(report.expected_dtype_changes_only)
                self.assertEqual(report.changed_cell_count, 3)
                self.assertEqual(report.newly_introduced_null_count, 0)

    def test_invalid_values_block_entire_column_even_at_99_percent(self):
        for values in (["12"] * 99 + ["bad"], ["12"] * 3 + ["bad"] * 2, [None]):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    compile_for(pd.DataFrame({"value": values}))

    def test_identifiers_classifications_unknown_and_unapproved_are_blocked(self):
        raw = pd.DataFrame({"value": ["12", "15"]})
        for proposal in (
            decision(kind="identifier"), decision(function="entity_reference"),
            decision(kind="unknown"), decision(function="unknown"),
            decision(kind="categorical", function="classification"),
            decision(requires_review=True), decision(risk_level="high"),
        ):
            with self.subTest(proposal=proposal):
                with self.assertRaises(ValueError):
                    compile_for(raw, proposal)

    def test_leading_zeros_and_ambiguous_symbols_rejected(self):
        for value in ("00123", "-00456", "+00001", "00.5", "1,234", "12%", "$12", "NaN", "Infinity", True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    compile_for(pd.DataFrame({"value": [value, "15"]}))

    def test_target_allow_list_integer_range_and_precision_loss(self):
        for values, target in (
            (["12.5"], "integer"), ([str(2**63)], "integer"),
            (["9007199254740993"], "float"), (["1e-400"], "float"),
            (["1e400"], "float"), (["1"], "bool"), (["1"], "float32"),
        ):
            with self.subTest(values=values, target=target):
                with self.assertRaises(ValueError):
                    compile_for(pd.DataFrame({"value": values}), decision(parameters={"target_dtype": target}))

    def test_integer_boundaries_and_existing_numeric_dtype_accounting(self):
        raw = pd.DataFrame({"value": [str(-(2**63)), str(2**63 - 1)]})
        cleaned, _ = execute_instruction(raw, compile_for(raw))
        self.assertEqual(cleaned.value.tolist(), [-(2**63), 2**63 - 1])
        raw = pd.DataFrame({"value": [12.0, 15.0]})
        cleaned, execution = execute_instruction(raw, compile_for(raw))
        self.assertEqual(execution.changed_cell_count, 2)
        self.assertTrue(execution.changed)

    def test_executor_rechecks_and_rolls_back_on_invalid_input(self):
        valid = pd.DataFrame({"value": ["12", "15"]})
        instruction = compile_for(valid)
        invalid = pd.DataFrame({"value": ["12", "invalid"]})
        cleaned, execution = execute_instruction(invalid, instruction)
        self.assertEqual(execution.status, "failed")
        self.assertEqual(execution.changed_cell_count, 0)
        pd.testing.assert_frame_equal(cleaned, invalid)

    def test_validation_rejects_wrong_numbers_dtype_nulls_or_unrelated_changes(self):
        raw = pd.DataFrame({"value": ["12", "15"], "other": [1, 2]})
        instruction = compile_for(raw)
        cleaned, _ = execute_instruction(raw, instruction)
        for field, value in (("value", 999), ("value", pd.NA), ("other", 42)):
            changed = cleaned.copy(deep=True)
            changed.loc[0, field] = value
            report, _ = validate_post_clean(raw, changed, profile_dataset(raw),
                modified_columns=["value"], executed_instructions=[instruction])
            self.assertFalse(report.passed)
        changed = cleaned.copy()
        changed["value"] = changed.value.astype("Float64")
        report, _ = validate_post_clean(raw, changed, profile_dataset(raw),
            modified_columns=["value"], executed_instructions=[instruction])
        self.assertFalse(report.expected_dtype_changes_only)

    def test_detector_candidate_is_not_a_license_to_coerce(self):
        raw = pd.DataFrame({"value": ["12", "15", "19", "invalid", "other"]})
        self.assertIn("mixed_numeric_representation_candidate", [i.issue_type for i in detect_issues(raw, profile_dataset(raw))])
        with self.assertRaises(ValueError):
            compile_for(raw)
        safe = pd.DataFrame({"value": ["12", "15", "19", "20", "21"]})
        cleaned, _ = execute_instruction(safe, compile_for(safe))
        self.assertNotIn("mixed_numeric_representation_candidate", [i.issue_type for i in detect_issues(cleaned, profile_dataset(cleaned))])

    def test_pipeline_trim_order_export_and_skip(self):
        for values, expected_executions in (([" 12 ", "15"] * 10, 1), (["12", "invalid"] * 10, 0)):
            with self.subTest(values=values[:2]), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "input.csv"
                pd.DataFrame({"value": values}).to_csv(source, index=False)
                raw_bytes = source.read_bytes()
                semantics = SemanticProfile.model_validate({
                    "dataset_context": {"row_entity": "record", "row_entity_confidence": 1.0},
                    "columns": [{"column_name": "value", "semantic_type": "numeric_measurement",
                                 "value_function": "measurement", "numeric_meaning": "magnitude",
                                 "concept_description": "Quantities",
                                 "confidence": 1.0, "reasoning_summary": "Test fixture"}],
                })
                def proposals(issues, _):
                    result = [decision()]
                    if expected_executions:
                        result.append(decision(decision_id="trim", proposed_action="trim_whitespace"))
                    return result, {}, []
                report = run_pipeline1(source, Path(directory) / "output",
                    semantic_analyzer=lambda *_: semantics, decision_engine=proposals)
                self.assertEqual(report.status, "success")
                self.assertEqual(report.decision_summary["execution_routing"]["executed_coerce_dtype_count"], expected_executions)
                self.assertEqual(source.read_bytes(), raw_bytes)
                if expected_executions:
                    self.assertEqual([i.action_id for i in report.instructions], ["trim_whitespace", "coerce_dtype"])
                else:
                    self.assertEqual(len(report.decision_summary["execution_routing"]["skipped_actions"]), 1)


if __name__ == "__main__":
    unittest.main()
