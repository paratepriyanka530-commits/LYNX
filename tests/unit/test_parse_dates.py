import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pipelines.pipeline1_cleaning.decisions.decision_schema import CleaningDecision
from pipelines.pipeline1_cleaning.orchestration.action_registry import ACTION_REGISTRY
from pipelines.pipeline1_cleaning.orchestration.action_validator import compile_instruction
from pipelines.pipeline1_cleaning.orchestration.data_cleaner_performer import execute_instruction
from pipelines.pipeline1_cleaning.pipeline import run_pipeline1
from pipelines.pipeline1_cleaning.profiling.dataset_profiler import profile_dataset
from pipelines.pipeline1_cleaning.semantics.batched_schema import SemanticProfile
from pipelines.pipeline1_cleaning.techniques.date_parsing import assess_dates
from pipelines.pipeline1_cleaning.validation.post_clean_validator import validate_post_clean


def proposal(kind="date", function="temporal", parameters=None, **overrides):
    fields = dict(
        decision_id="date-decision", source_issue_ids=["date-issue"],
        target_columns=["value"], decision_type="action", proposed_action="parse_dates",
        parameters=parameters or {}, confidence=0.98, rationale="Approved temporal values.",
        semantic_basis={"value": {"semantic_type": kind, "value_function": function}},
        risk_level="low", requires_review=False, decision_source="deterministic_tier1",
    )
    fields.update(overrides)
    return CleaningDecision(**fields)


def compile_for(frame, decision=None):
    return compile_instruction(decision or proposal(), list(frame.columns), dict(frame.dtypes), dataframe=frame)


class ParseDatesTests(unittest.TestCase):
    def test_safe_dates_preserve_nulls_raw_unrelated_and_lineage(self):
        raw = pd.DataFrame({"value": pd.Series(
            ["2025-01-31", "2025/01/31", "31/01/2025", None, pd.NA, float("nan")], dtype=object
        ), "other": [" x "] * 6})
        snapshot = raw.copy(deep=True)
        instruction = compile_for(raw)
        cleaned, execution = execute_instruction(raw, instruction)
        self.assertEqual(cleaned.value.iloc[:3].tolist(), ["2025-01-31"] * 3)
        self.assertIsNone(cleaned.value.iloc[3])
        pd.testing.assert_series_equal(raw.value.isna(), cleaned.value.isna())
        pd.testing.assert_frame_equal(raw, snapshot)
        pd.testing.assert_series_equal(raw.other, cleaned.other)
        self.assertEqual(execution.changed_cell_count, 2)
        self.assertEqual(execution.source_issue_ids, ["date-issue"])
        self.assertEqual(execution.source_decision_id, "date-decision")
        self.assertEqual(instruction.parameters, {"output_format": "YYYY-MM-DD"})

    def test_ambiguous_dates_block_even_above_coverage_threshold(self):
        frame = pd.DataFrame({"value": ["2025/01/31"] * 99 + ["01/02/2025"]})
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            compile_for(frame)

    def test_low_coverage_and_empty_column_rejected(self):
        for values in (["2025-01-31"] * 18 + ["bad", "2025-02-30"], [None]):
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, "below 95%"):
                    compile_for(pd.DataFrame({"value": values}))

    def test_95_percent_coverage_preserves_unparseable_value(self):
        frame = pd.DataFrame({"value": ["2025/01/31"] * 19 + ["not a date"]})
        instruction = compile_for(frame)
        cleaned, result = execute_instruction(frame, instruction)
        self.assertEqual(cleaned.value.iloc[-1], "not a date")
        self.assertEqual(cleaned.value.isna().sum(), 0)
        self.assertEqual(result.changed_cell_count, 19)

    def test_temporal_semantics_and_approval_required(self):
        frame = pd.DataFrame({"value": ["2025/01/31"]})
        for decision in (
            proposal(kind="categorical"), proposal(function="classification"),
            proposal(requires_review=True), proposal(risk_level="high"),
        ):
            with self.subTest(decision=decision):
                with self.assertRaises(ValueError):
                    compile_for(frame, decision)

    def test_allow_list_and_target_enforced(self):
        frame = pd.DataFrame({"value": ["2025/01/31"]})
        for decision in (
            proposal(parameters={"output_format": "%d/%m/%Y"}),
            proposal(parameters={"output_format": "HH:MM:SS"}),
            proposal(parameters={"dayfirst": True}), proposal(target_columns=[]),
            proposal(target_columns=["missing"]),
        ):
            with self.subTest(decision=decision):
                with self.assertRaises(ValueError):
                    compile_for(frame, decision)

    def test_datetime_and_time_are_lossless_supported_forms(self):
        for kind, values, expected in (
            ("datetime", ["2025/01/31 12:30", "2025-01-31T12:30:45"],
             ["2025-01-31 12:30:00", "2025-01-31 12:30:45"]),
            ("time", ["12:30", "23:59:59"], ["12:30:00", "23:59:59"]),
        ):
            with self.subTest(kind=kind):
                frame = pd.DataFrame({"value": values})
                cleaned, _ = execute_instruction(frame, compile_for(frame, proposal(kind=kind)))
                self.assertEqual(cleaned.value.tolist(), expected)
        for value in ("2025-01-31T12:30:45Z", "2025-01-31 12:30:45.123"):
            with self.assertRaises(ValueError):
                compile_for(pd.DataFrame({"value": [value]}), proposal(kind="datetime"))

    def test_executor_rechecks_actual_coverage(self):
        valid = pd.DataFrame({"value": ["2025-01-31"]})
        instruction = compile_for(valid)
        ambiguous = pd.DataFrame({"value": ["01/02/2025"]})
        cleaned, execution = execute_instruction(ambiguous, instruction)
        self.assertEqual(execution.status, "failed")
        pd.testing.assert_frame_equal(cleaned, ambiguous)

    def test_validation_accepts_cardinality_collapse_and_rejects_tampering(self):
        raw = pd.DataFrame({"value": ["2025/01/31", "31/01/2025"] * 10, "other": range(20)})
        instruction = compile_for(raw)
        cleaned, _ = execute_instruction(raw, instruction)
        kwargs = dict(before_profile=profile_dataset(raw), modified_columns=["value"], executed_instructions=[instruction])
        report, _ = validate_post_clean(raw, cleaned, **kwargs)
        self.assertTrue(report.passed)
        self.assertTrue(report.targeted_date_formats_improved)
        self.assertFalse(report.cardinality_equal)
        self.assertEqual(report.newly_introduced_null_count, 0)
        cleaned.loc[0, "value"] = "2030-01-31"
        report, _ = validate_post_clean(raw, cleaned, **kwargs)
        self.assertFalse(report.only_approved_date_changes)
        self.assertFalse(report.passed)
        cleaned.loc[0, "value"] = None
        report, _ = validate_post_clean(raw, cleaned, **kwargs)
        self.assertEqual(report.newly_introduced_null_count, 1)

    def test_pipeline_executes_approved_date_and_skips_ambiguous_proposal(self):
        for values, expected_count in ((["2025/01/31", "31/01/2025"] * 10, 1), (["01/02/2025"] * 20, 0)):
            with self.subTest(expected_count=expected_count), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "input.csv"
                pd.DataFrame({"value": values}).to_csv(path, index=False)
                raw_bytes = path.read_bytes()
                semantics = SemanticProfile.model_validate({
                    "dataset_context": {"row_entity": "record", "row_entity_confidence": 1.0},
                    "columns": [{"column_name": "value", "semantic_type": "date", "value_function": "temporal",
                                 "numeric_meaning": "not_applicable", "concept_description": "Calendar dates",
                                 "confidence": 1.0, "reasoning_summary": "Test fixture"}],
                })
                report = run_pipeline1(path, Path(directory) / "output",
                    semantic_analyzer=lambda *_: semantics,
                    decision_engine=lambda issues, _: ([proposal(source_issue_ids=[issues[0].issue_id] if issues else ["test-issue"])], {}, []))
                self.assertEqual(report.status, "success")
                self.assertEqual(report.decision_summary["execution_routing"]["executed_parse_dates_count"], expected_count)
                self.assertEqual(path.read_bytes(), raw_bytes)
                if not expected_count:
                    self.assertIn("Ambiguous", report.decision_summary["execution_routing"]["skipped_actions"][0]["reason"])


if __name__ == "__main__":
    unittest.main()
