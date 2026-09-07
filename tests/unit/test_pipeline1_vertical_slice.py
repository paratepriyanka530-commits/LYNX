import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
from pydantic import ValidationError

from pipelines.pipeline1_cleaning.decisions.decision_schema import CleaningDecision
from pipelines.pipeline1_cleaning.orchestration.action_registry import get_action
from pipelines.pipeline1_cleaning.orchestration.action_validator import compile_instruction
from pipelines.pipeline1_cleaning.orchestration.data_cleaner_performer import execute_instruction
from pipelines.pipeline1_cleaning.profiling.dataset_profiler import profile_dataset
from pipelines.pipeline1_cleaning.validation.post_clean_validator import validate_post_clean
from pipelines.pipeline1_cleaning.pipeline import _run_default_semantic_analyzer


class Pipeline1VerticalSliceUnitTests(unittest.TestCase):
    @patch("pipelines.pipeline1_cleaning.pipeline.analyze_batched_semantics")
    @patch("intelligence.rag.retriever.RAGRetriever")
    @patch("intelligence.llm.ollama_provider.OllamaProvider")
    def test_default_semantic_runtime_indexes_and_returns_analysis(
        self, provider_class, retriever_class, analyze
    ):
        expected = Mock()
        analyze.return_value = expected
        result = _run_default_semantic_analyzer({"profile": True}, {"evidence": True})
        retriever_class.return_value.index_knowledge.assert_called_once_with()
        self.assertIs(result, expected)

    def test_no_op_executes_on_copy_and_validates_identically(self):
        original = pd.DataFrame({"name": [" A ", None], "value": [1, 2]})
        snapshot = original.copy(deep=True)
        decision = CleaningDecision(
            decision_id="d1", source_issue_ids=["i1"], decision_type="no_action",
            proposed_action="no_op", confidence=1.0, rationale="Test no-op.",
            risk_level="low", requires_review=False,
            decision_source="deterministic_tier1",
        )
        instruction = compile_instruction(decision, list(original.columns))
        cleaned, execution = execute_instruction(original, instruction)
        validation, _ = validate_post_clean(
            original, cleaned, profile_dataset(original), expect_no_changes=True
        )

        self.assertIsNot(cleaned, original)
        pd.testing.assert_frame_equal(original, snapshot)
        pd.testing.assert_frame_equal(cleaned, original)
        self.assertFalse(execution.changed)
        self.assertTrue(validation.passed)
        self.assertTrue(validation.dataframe_equal)

    def test_unknown_action_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown action_id"):
            get_action("not_registered")

    def test_compiler_rejects_unknown_action(self):
        with self.assertRaises(ValidationError):
            CleaningDecision(
                decision_id="d1", source_issue_ids=["i1"], decision_type="action",
                proposed_action="not_registered", confidence=1.0,
                rationale="Test invalid action.", risk_level="low",
                requires_review=False, decision_source="deterministic_tier1",
            )

    def test_trim_whitespace_is_targeted_and_preserves_values(self):
        original = pd.DataFrame({
            "target": ["  Sales", "Sales   ", "  New York ", "New   York", None, 7, "   "],
            "other": [" keep ", "same", "x", "y", "z", "n", "w"],
        })
        snapshot = original.copy(deep=True)
        decision = CleaningDecision(
            decision_id="d-trim", source_issue_ids=["i-whitespace"],
            target_columns=["target"], decision_type="action",
            proposed_action="trim_whitespace", confidence=0.98,
            rationale="Deterministic edge trimming.", risk_level="low",
            requires_review=False, decision_source="deterministic_tier1",
        )
        instruction = compile_instruction(decision, list(original.columns))
        cleaned, execution = execute_instruction(original, instruction)

        self.assertEqual(
            cleaned["target"].tolist(),
            ["Sales", "Sales", "New York", "New   York", None, 7, ""],
        )
        self.assertEqual(cleaned["other"].tolist(), original["other"].tolist())
        pd.testing.assert_frame_equal(original, snapshot)
        self.assertEqual(instruction.source_issue_ids, ["i-whitespace"])
        self.assertEqual(execution.source_decision_id, "d-trim")
        self.assertEqual(execution.source_issue_ids, ["i-whitespace"])
        self.assertEqual(execution.target_columns, ["target"])
        self.assertEqual(execution.changed_cell_count, 4)

    def test_flag_and_unapproved_action_do_not_compile(self):
        flagged = CleaningDecision(
            decision_id="d-flag", source_issue_ids=["i1"],
            target_columns=["value"], decision_type="flag_for_review",
            proposed_action="flag_rows", confidence=0.9, rationale="Review.",
            risk_level="high", requires_review=True,
            decision_source="deterministic_tier1",
        )
        with self.assertRaisesRegex(ValueError, "Unknown action_id"):
            compile_instruction(flagged, ["value"])

    def test_post_clean_validation_confirms_targeted_whitespace_improvement(self):
        original = pd.DataFrame({"target": [" a ", "b"], "other": [" x ", "y"]})
        decision = CleaningDecision(
            decision_id="d1", source_issue_ids=["i1"], target_columns=["target"],
            decision_type="action", proposed_action="trim_whitespace",
            confidence=1.0, rationale="Trim.", risk_level="low",
            requires_review=False, decision_source="deterministic_tier1",
        )
        cleaned, _ = execute_instruction(
            original, compile_instruction(decision, list(original.columns))
        )
        validation, _ = validate_post_clean(
            original, cleaned, profile_dataset(original), modified_columns=["target"]
        )
        self.assertTrue(validation.passed)
        self.assertTrue(validation.targeted_whitespace_improved)
        self.assertTrue(validation.unrelated_columns_equal)
        self.assertEqual(validation.changed_cell_count, 1)

    def test_explicit_categorical_mapping_is_exact_and_preserving(self):
        original = pd.DataFrame({
            "segment": ["sales", "Sales", "sales team", None],
            "other": ["x", "y", "z", "w"],
        })
        snapshot = original.copy(deep=True)
        decision = CleaningDecision(
            decision_id="d-normalize", source_issue_ids=["i-case"],
            target_columns=["segment"], decision_type="action",
            proposed_action="normalize_categorical_labels",
            parameters={"mapping": {"sales": "Sales"}}, confidence=1.0,
            rationale="Approved exact mapping.", risk_level="low",
            requires_review=False, decision_source="llm_tier2",
        )
        instruction = compile_instruction(
            decision, list(original.columns), dict(original.dtypes)
        )
        cleaned, execution = execute_instruction(original, instruction)

        self.assertEqual(cleaned["segment"].iloc[:3].tolist(), ["Sales", "Sales", "sales team"])
        self.assertTrue(pd.isna(original["segment"].iloc[3]))
        self.assertTrue(pd.isna(cleaned["segment"].iloc[3]))
        self.assertEqual(cleaned["other"].tolist(), original["other"].tolist())
        pd.testing.assert_frame_equal(original, snapshot)
        self.assertEqual(execution.changed_cell_count, 1)
        self.assertEqual(execution.source_issue_ids, ["i-case"])

    def test_categorical_mapping_is_required_and_must_be_nonempty(self):
        for parameters in ({}, {"mapping": {}}):
            with self.subTest(parameters=parameters):
                decision = CleaningDecision(
                    decision_id="d1", source_issue_ids=["i1"],
                    target_columns=["segment"], decision_type="action",
                    proposed_action="normalize_categorical_labels",
                    parameters=parameters, confidence=1.0, rationale="Invalid.",
                    risk_level="low", requires_review=False,
                    decision_source="llm_tier2",
                )
                with self.assertRaisesRegex(ValueError, "non-empty mapping"):
                    compile_instruction(decision, ["segment"], {"segment": object})

    def test_categorical_mapping_rejects_non_string_labels(self):
        decision = CleaningDecision(
            decision_id="d1", source_issue_ids=["i1"], target_columns=["segment"],
            decision_type="action", proposed_action="normalize_categorical_labels",
            parameters={"mapping": {1: "one"}}, confidence=1.0,
            rationale="Invalid.", risk_level="low", requires_review=False,
            decision_source="llm_tier2",
        )
        with self.assertRaisesRegex(ValueError, "must be strings"):
            compile_instruction(decision, ["segment"], {"segment": object})


if __name__ == "__main__":
    unittest.main()
