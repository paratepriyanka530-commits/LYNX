import json
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
from pipelines.pipeline1_cleaning.techniques.missing_values import (
    assess_median_semantic_eligibility,
    missingness_assessment,
)
from pipelines.pipeline1_cleaning.validation.post_clean_validator import validate_post_clean


def proposal(strategy="median", kind="numeric_measurement", function="measurement", evidence=None,
             parameters=None, semantic_confidence=0.99, **overrides):
    data = dict(
        decision_id="imputation-decision", source_issue_ids=["missing-issue"],
        target_columns=["value"], decision_type="action", proposed_action="impute_missing",
        parameters=parameters if parameters is not None else {"strategy": strategy},
        semantic_basis={"value": {"semantic_type": kind, "value_function": function,
                                  "confidence": semantic_confidence}},
        evidence_summary=evidence or {}, confidence=0.99,
        rationale="Explicitly approved strategy.", risk_level="low", requires_review=False,
        decision_source="deterministic_tier1",
    )
    data.update(overrides)
    return CleaningDecision(**data)


def frame(values=None, dtype="float64"):
    values = values if values is not None else [10] * 20 + [20] * 19 + [None]
    return pd.DataFrame({"value": pd.Series(values, dtype=dtype), "other": range(len(values))})


def compile_for(raw, decision):
    return compile_instruction(decision, list(raw.columns), dict(raw.dtypes), dataframe=raw)


class ImputeMissingTests(unittest.TestCase):
    def execute_and_validate(self, raw, decision):
        snapshot = raw.copy(deep=True)
        instruction = compile_for(raw, decision)
        cleaned, execution = execute_instruction(raw, instruction)
        self.assertEqual(execution.status, "success")
        report, _ = validate_post_clean(raw, cleaned, profile_dataset(raw),
            modified_columns=["value"], executed_instructions=[instruction])
        self.assertTrue(report.passed, report.warnings)
        pd.testing.assert_frame_equal(raw, snapshot)
        pd.testing.assert_series_equal(raw.other, cleaned.other)
        pd.testing.assert_series_equal(raw.value[raw.value.notna()], cleaned.value[raw.value.notna()])
        self.assertEqual(raw.shape, cleaned.shape)
        self.assertEqual(list(raw.columns), list(cleaned.columns))
        self.assertEqual(report.newly_introduced_null_count, 0)
        self.assertTrue(report.expected_missingness_changes_only)
        self.assertEqual(report.imputed_cell_count, int(raw.value.isna().sum()))
        self.assertEqual(execution.changed_cell_count, report.imputed_cell_count)
        self.assertEqual(execution.source_issue_ids, ["missing-issue"])
        self.assertEqual(execution.source_decision_id, "imputation-decision")
        self.assertLessEqual(len(execution.change_examples), 5)
        return cleaned, execution, report

    def test_low_missing_measurement_median_computed_from_observations(self):
        for dtype in ("float64", "Float64", "Int64"):
            with self.subTest(dtype=dtype):
                cleaned, execution, _ = self.execute_and_validate(frame(dtype=dtype), proposal())
                self.assertEqual(cleaned.value.iloc[-1], 10)
                self.assertEqual(execution.audit_metadata["replacement_value"], 10)
                self.assertEqual(execution.audit_metadata["missing_percentage"], 2.5)
                self.assertEqual(execution.audit_metadata["strategy"], "median")
                self.assertEqual(str(cleaned.value.dtype), dtype)

    def test_statistics_cannot_be_supplied_by_model(self):
        for parameters in ({"strategy": "median", "value": 999}, {"strategy": "median", "median": 999},
                           {"strategy": "mode", "replacement": "invented"}, {"strategy": "mean"}):
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    compile_for(frame(), proposal(parameters=parameters))

    def test_missingness_boundaries_and_minimum_observations(self):
        for missing, expected in ((0, "none"), (5, "low"), (6, "moderate"), (20, "moderate"),
                                  (21, "high"), (50, "high"), (51, "very_high")):
            raw = frame([10] * (100 - missing) + [None] * missing)
            self.assertEqual(missingness_assessment(raw.value)["missingness_risk"], expected)
            if expected == "low":
                compile_for(raw, proposal())
            else:
                with self.assertRaises(ValueError):
                    compile_for(raw, proposal())
        with self.assertRaisesRegex(ValueError, "20 observed"):
            compile_for(frame([10] * 19 + [None]), proposal())

    def test_prohibited_or_uncertain_semantics_block_even_tiny_missingness(self):
        raw = frame([10] * 999 + [None])
        for kind, function in (("identifier", "measurement"), ("numeric_measurement", "entity_reference"),
                               ("unknown", "measurement"), ("numeric_measurement", "unknown"),
                               ("date", "temporal"), ("time", "temporal"), ("datetime", "temporal"),
                               ("text", "descriptive_attribute"), ("contact_information", "contact"),
                               ("location", "location")):
            with self.subTest(kind=kind, function=function):
                with self.assertRaises(ValueError):
                    compile_for(raw, proposal(kind=kind, function=function))
        for confidence in (None, 0.2, 0.79):
            with self.assertRaises(ValueError):
                compile_for(raw, proposal(semantic_confidence=confidence))
        with self.assertRaises(ValueError):
            compile_for(frame(["00123"] * 39 + [None], dtype=object), proposal())
        transaction = proposal()
        transaction.semantic_basis["value"]["semantic_role"] = "transaction_identifier"
        with self.assertRaises(ValueError):
            compile_for(raw, transaction)
        with self.assertRaises(ValueError):
            compile_for(raw, proposal(evidence={"identifier_like": True}))

    def test_median_integer_fraction_and_nonfinite_safety(self):
        raw = frame([1] * 20 + [2] * 20 + [None], dtype="Int64")
        with self.assertRaisesRegex(ValueError, "Non-integer median"):
            compile_for(raw, proposal())
        cleaned, execution, _ = self.execute_and_validate(raw.astype({"value": "Float64"}), proposal())
        self.assertEqual(execution.audit_metadata["replacement_value"], 1.5)
        self.assertEqual(cleaned.value.iloc[-1], 1.5)
        with self.assertRaisesRegex(ValueError, "finite"):
            compile_for(frame([float("inf")] + [1] * 38 + [None]), proposal())

    def test_categorical_constants_require_explicit_meaning(self):
        raw = frame(["A"] * 39 + [None], dtype=object)
        for constant_type, expected in (("unknown", "Unknown"), ("not_available", "Not Available"),
                                        ("not_applicable", "Not Applicable")):
            decision = proposal(kind="categorical", function="classification",
                parameters={"strategy": "constant", "constant_type": constant_type},
                evidence={"approved_missingness_interpretation": constant_type,
                          "missingness_justification": "Source documentation explicitly defines this missing state."})
            cleaned, execution, _ = self.execute_and_validate(raw, decision)
            self.assertEqual(cleaned.value.iloc[-1], expected)
            self.assertEqual(execution.audit_metadata["replacement_value"], expected)
        for parameters in ({"strategy": "constant", "constant_type": "invented"},
                           {"strategy": "constant", "replacement": "Unknown"},
                           {"strategy": "constant", "constant_type": "unknown"}):
            with self.assertRaises(ValueError):
                compile_for(raw, proposal(kind="categorical", function="classification", parameters=parameters))

    def test_unique_mode_is_calculated_and_ties_rejected(self):
        evidence = {"missingness_is_meaningful": False,
                    "missingness_justification": "Confirmed acquisition omissions; classification remains applicable."}
        decision = proposal(strategy="mode", kind="categorical", function="classification", evidence=evidence)
        raw = frame(["A"] * 20 + ["B"] * 19 + [None], dtype=object)
        cleaned, execution, _ = self.execute_and_validate(raw, decision)
        self.assertEqual(cleaned.value.iloc[-1], "A")
        self.assertEqual(execution.audit_metadata["replacement_value"], "A")
        for values in (["A"] * 10 + ["B"] * 10 + [None], ["A"] * 30 + [None] * 10):
            with self.assertRaises(ValueError):
                compile_for(frame(values, dtype=object), decision)
        for bad in (proposal(strategy="mode", kind="categorical", function="classification"),
                    proposal(strategy="mode", kind="unknown", function="classification", evidence=evidence)):
            with self.assertRaises(ValueError):
                compile_for(raw, bad)

    def test_boolean_defaults_preserve_null_and_explicit_state_is_required(self):
        raw = frame([True] * 39 + [None], dtype="boolean")
        params = {"strategy": "constant", "constant_type": "false"}
        for decision in (proposal(kind="boolean", function="boolean_state", parameters=params),
                         proposal(kind="boolean", function="boolean_state", strategy="mode")):
            with self.assertRaises(ValueError):
                compile_for(raw, decision)
        for state in ("true", "false"):
            decision = proposal(kind="boolean", function="boolean_state",
                parameters={"strategy": "constant", "constant_type": state},
                evidence={"approved_missingness_interpretation": state,
                          "missingness_justification": "Documented source encoding: NULL denotes this exact state."})
            cleaned, _, _ = self.execute_and_validate(raw, decision)
            self.assertEqual(bool(cleaned.value.iloc[-1]), state == "true")
        with self.assertRaises(ValueError):
            compile_for(raw, decision.model_copy(update={"requires_review": True}))

    def test_executor_rechecks_missingness_and_preserves_on_failure(self):
        instruction = compile_for(frame(), proposal())
        changed_input = frame([10] * 30 + [None] * 10)
        cleaned, result = execute_instruction(changed_input, instruction)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.audit_metadata["strategy"], "median")
        self.assertEqual(result.audit_metadata["missing_percentage"], 25.0)
        pd.testing.assert_frame_equal(cleaned, changed_input)

    def test_validation_detects_wrong_replacement_observed_mutation_and_new_nulls(self):
        raw = frame()
        instruction = compile_for(raw, proposal())
        cleaned, _ = execute_instruction(raw, instruction)
        for index, value in ((39, 999), (0, 999), (0, None), (39, None)):
            corrupted = cleaned.copy(deep=True)
            corrupted.loc[index, "value"] = value
            report, _ = validate_post_clean(raw, corrupted, profile_dataset(raw),
                modified_columns=["value"], executed_instructions=[instruction])
            self.assertFalse(report.passed)

    def test_pipeline_order_lineage_and_no_schema_expansion(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "raw.csv"
            frame().to_csv(source, index=False)
            raw_bytes = source.read_bytes()
            semantic = SemanticProfile.model_validate({
                "dataset_context": {"row_entity": "record", "row_entity_confidence": 1.0},
                "columns": [{"column_name": name, "semantic_type": "numeric_measurement", "value_function": "measurement",
                             "numeric_meaning": "magnitude",
                             "concept_description": "Confirmed quantities", "confidence": 0.99,
                             "reasoning_summary": "Approved test evidence"} for name in ("value", "other")],
            })
            def decisions(issues, _):
                missing_id = next(item.issue_id for item in issues if item.issue_type == "missing_values")
                return [proposal(source_issue_ids=[missing_id]),
                        proposal(proposed_action="coerce_dtype", decision_id="coerce", source_issue_ids=[missing_id],
                                 parameters={"target_dtype": "integer"})], {}, []
            output = Path(directory) / "output"
            report = run_pipeline1(source, output, semantic_analyzer=lambda *_: semantic, decision_engine=decisions)
            self.assertEqual(report.status, "success", report.warnings)
            self.assertEqual([item.action_id for item in report.instructions], ["coerce_dtype", "impute_missing"])
            self.assertEqual(report.validation.imputed_cell_count, 1)
            self.assertEqual(report.validation.after_summary["total_missing_cells"], 0)
            self.assertEqual(source.read_bytes(), raw_bytes)
            exported = pd.read_csv(output / "cleaned_dataset.csv")
            self.assertEqual(exported.shape, (40, 2))
            self.assertEqual(list(exported.columns), ["value", "other"])
            saved = json.loads((output / "cleaning_report.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["executions"][-1]["audit_metadata"]["replacement_value"], 10)
            self.assertEqual(len(ACTION_REGISTRY), 7)
            self.assertIn("mask_column", ACTION_REGISTRY)
            self.assertNotIn("add_missing_indicator", ACTION_REGISTRY)


class MedianSemanticEligibilityTests(unittest.TestCase):
    @staticmethod
    def semantic(numeric_meaning="magnitude", kind="numeric_measurement",
                 function="measurement", confidence=0.9, role="unknown"):
        return {
            "semantic_type": kind,
            "value_function": function,
            "numeric_meaning": numeric_meaning,
            "semantic_role": role,
            "confidence": confidence,
            "concept_description": "A numeric concept used by the regression fixture.",
        }

    @staticmethod
    def repeated_measurement(values):
        return pd.Series(values + [None], dtype="float64")

    def assess(self, values, **semantic):
        return assess_median_semantic_eligibility(
            self.semantic(**semantic), values,
        )

    def test_five_regression_measurements_are_semantically_eligible(self):
        fixtures = {
            "total_amount": self.repeated_measurement([10.5, 20.5, 30.5] * 20),
            "attendance_percent": self.repeated_measurement([55.2, 70.4, 85.6] * 20),
            "actual_quantity": self.repeated_measurement([100, 200, 300] * 20),
            "defect_quantity": self.repeated_measurement([0, 1, 2] * 20),
            "humidity_percent": self.repeated_measurement([45.5, 55.0, 65.5] * 20),
        }
        for column, series in fixtures.items():
            with self.subTest(column=column):
                result = self.assess(series)
                self.assertEqual(result["semantic_eligible"], "true")
                self.assertEqual(result["reason_code"], "confirmed_quantitative_magnitude")

    def test_source_port_shape_is_never_automatically_authorized(self):
        series = pd.Series(list(range(1000, 1100)) + [None], dtype="float64")
        result = self.assess(series)
        self.assertEqual(result["semantic_eligible"], "unknown")
        self.assertEqual(result["reason_code"], "high_cardinality_integral_label_risk")

    def test_explicit_non_magnitude_meanings_are_blocked(self):
        series = self.repeated_measurement([1, 2, 3] * 20)
        for meaning, reason in (
            ("nominal_code", "nominal_numeric_code"),
            ("identifier", "numeric_identifier"),
            ("ordinal", "ordered_values_are_not_magnitudes"),
            ("not_applicable", "numeric_meaning_not_applicable"),
        ):
            with self.subTest(meaning=meaning):
                result = self.assess(series, numeric_meaning=meaning)
                self.assertEqual(result["semantic_eligible"], "false")
                self.assertEqual(result["reason_code"], reason)

    def test_identifier_and_entity_reference_evidence_block(self):
        series = self.repeated_measurement([1, 2, 3] * 20)
        cases = (
            {"kind": "identifier"},
            {"function": "entity_reference"},
            {"role": "foreign_identifier"},
        )
        for case in cases:
            with self.subTest(case=case):
                result = self.assess(series, **case)
                self.assertEqual(result["semantic_eligible"], "false")
                self.assertEqual(result["reason_code"], "identifier_or_entity_reference")
        result = assess_median_semantic_eligibility(
            self.semantic(), series, {"identifier_like": True},
        )
        self.assertEqual(result["semantic_eligible"], "false")

    def test_unknown_low_confidence_and_contradictions_fail_closed(self):
        series = self.repeated_measurement([1.5, 2.5, 3.5] * 20)
        cases = (
            ({"numeric_meaning": "unknown"}, "numeric_meaning_unknown"),
            ({"numeric_meaning": None}, "numeric_meaning_unknown"),
            ({"confidence": 0.79}, "insufficient_semantic_confidence"),
            ({"kind": "categorical"}, "contradictory_quantitative_semantics"),
            ({"function": "classification"}, "contradictory_quantitative_semantics"),
        )
        for changes, reason in cases:
            with self.subTest(changes=changes):
                result = self.assess(series, **changes)
                self.assertEqual(result["semantic_eligible"], "unknown")
                self.assertEqual(result["reason_code"], reason)

    def test_numeric_label_heuristic_is_veto_only(self):
        high_cardinality = pd.Series(list(range(100)) + [None], dtype="float64")
        result = self.assess(high_cardinality, numeric_meaning="unknown")
        self.assertEqual(result["semantic_eligible"], "unknown")
        self.assertEqual(result["reason_code"], "numeric_meaning_unknown")
        nominal_repeated = self.repeated_measurement([1, 2, 3] * 20)
        result = self.assess(nominal_repeated, numeric_meaning="nominal_code")
        self.assertEqual(result["semantic_eligible"], "false")


class Tier2MedianAuthorizationTests(unittest.TestCase):
    @staticmethod
    def semantic_profile(columns, meanings=None, confidence=0.99):
        meanings = meanings or {}
        return SemanticProfile.model_validate({
            "dataset_context": {"row_entity": "record", "row_entity_confidence": 1.0},
            "columns": [{
                "column_name": column,
                "semantic_type": meanings.get(column, {}).get("semantic_type", "numeric_measurement"),
                "value_function": meanings.get(column, {}).get("value_function", "measurement"),
                "numeric_meaning": meanings.get(column, {}).get("numeric_meaning", "magnitude"),
                "concept_description": meanings.get(column, {}).get(
                    "concept_description", "A quantitative value associated with the record."
                ),
                "semantic_role": "unknown",
                "confidence": meanings.get(column, {}).get("confidence", confidence),
                "reasoning_summary": "The fixture supplies the semantic contract under test.",
            } for column in columns],
        })

    @staticmethod
    def tier2_median_decisions(issues, _semantic_profile):
        return [CleaningDecision(
            decision_id=f"tier2-median-{issue.target_columns[0]}",
            source_issue_ids=[issue.issue_id],
            target_columns=[issue.target_columns[0]],
            decision_type="action",
            proposed_action="impute_missing",
            parameters={"strategy": "median"},
            confidence=0.91,
            rationale="The model recommends median imputation for deterministic adjudication.",
            evidence_summary={},
            semantic_basis={},
            risk_level="high",
            requires_review=True,
            decision_source="llm_tier2",
        ) for issue in issues if issue.issue_type == "missing_values"], {}, []

    def run_fixture(self, raw, meanings=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        source = root / "raw.csv"
        raw.to_csv(source, index=False)
        raw_bytes = source.read_bytes()
        profile = self.semantic_profile(list(raw.columns), meanings)
        report = run_pipeline1(
            source,
            root / "output",
            semantic_analyzer=lambda *_: profile,
            decision_engine=self.tier2_median_decisions,
        )
        exported = pd.read_csv(root / "output" / "cleaned_dataset.csv")
        self.assertEqual(source.read_bytes(), raw_bytes)
        return report, exported, source, raw_bytes

    def test_five_positive_regressions_authorize_compile_and_execute(self):
        positive_columns = (
            "total_amount", "attendance_percent", "actual_quantity",
            "defect_quantity", "humidity_percent",
        )
        raw = pd.DataFrame({
            "total_amount": [10.5, 20.5] * 20 + [None],
            "attendance_percent": [55.2, 85.6] * 20 + [None],
            "actual_quantity": [100.0, 200.0] * 20 + [None],
            "defect_quantity": [1.0, 2.0] * 20 + [None],
            "humidity_percent": [45.5, 65.5] * 20 + [None],
            # Deliberately preserve the currently mistaken magnitude claim; the
            # deterministic numeric-label-risk veto must still block authorization.
            "source_port": list(range(1000, 1040)) + [None],
            "row_marker": list(range(41)),
        })
        snapshot = raw.copy(deep=True)
        report, cleaned, _, _ = self.run_fixture(raw)

        authorization = {
            item["target_columns"][0]: item
            for item in report.decision_summary["execution_routing"]
            ["median_imputation_authorization_assessments"]
        }
        for column in positive_columns:
            with self.subTest(column=column):
                assessment = authorization[column]
                self.assertEqual(assessment["llm_recommendation"]["risk_level"], "high")
                self.assertTrue(assessment["llm_recommendation"]["requires_review"])
                self.assertEqual(assessment["semantic_eligibility"]["semantic_eligible"], "true")
                self.assertEqual(assessment["mechanical_eligibility"]["status"], "passed")
                self.assertTrue(assessment["final_adjudication"]["authorized"])
                decision = next(item for item in report.decisions if item.target_columns == [column])
                self.assertEqual(decision.risk_level, "low")
                self.assertFalse(decision.requires_review)
                self.assertTrue(cleaned[column].notna().all())
                pd.testing.assert_series_equal(
                    cleaned.loc[raw[column].notna(), column].reset_index(drop=True),
                    raw.loc[raw[column].notna(), column].reset_index(drop=True),
                    check_names=False,
                )

        source_assessment = authorization["source_port"]
        self.assertEqual(source_assessment["semantic_eligibility"]["semantic_eligible"], "unknown")
        self.assertEqual(
            source_assessment["semantic_eligibility"]["reason_code"],
            "high_cardinality_integral_label_risk",
        )
        self.assertEqual(source_assessment["mechanical_eligibility"]["status"], "not_evaluated")
        self.assertFalse(source_assessment["final_adjudication"]["authorized"])
        source_decision = next(item for item in report.decisions if item.target_columns == ["source_port"])
        self.assertEqual(source_decision.risk_level, "high")
        self.assertTrue(source_decision.requires_review)
        self.assertTrue(pd.isna(cleaned["source_port"].iloc[-1]))
        self.assertFalse(any(
            item.action_id == "impute_missing" and item.target_columns == ["source_port"]
            for item in report.instructions
        ))
        semantic_skip = next(
            item for item in report.decision_summary["execution_routing"]["skipped_actions"]
            if item["target_columns"] == ["source_port"]
        )
        self.assertEqual(semantic_skip["rejection_stage"], "semantic_eligibility")
        self.assertEqual(semantic_skip["reason_code"], "high_cardinality_integral_label_risk")
        pd.testing.assert_frame_equal(raw, snapshot)

    def test_forced_source_port_recommendation_preserves_entire_dataframe(self):
        raw = pd.DataFrame({
            "source_port": list(range(1000, 1040)) + [None],
            "row_marker": list(range(41)),
        })
        report, cleaned, _, _ = self.run_fixture(raw)
        assessment = report.decision_summary["execution_routing"] \
            ["median_imputation_authorization_assessments"][0]
        self.assertEqual(assessment["llm_recommendation"]["proposed_action"], "impute_missing")
        self.assertEqual(assessment["llm_recommendation"]["parameters"], {"strategy": "median"})
        self.assertEqual(assessment["llm_recommendation"]["risk_level"], "high")
        self.assertTrue(assessment["llm_recommendation"]["requires_review"])
        self.assertEqual(assessment["semantic_eligibility"]["semantic_eligible"], "unknown")
        self.assertEqual(
            assessment["semantic_eligibility"]["reason_code"],
            "high_cardinality_integral_label_risk",
        )
        self.assertFalse(assessment["final_adjudication"]["authorized"])
        self.assertFalse(any(item.action_id == "impute_missing" for item in report.instructions))
        pd.testing.assert_frame_equal(cleaned, raw, check_dtype=False)

    def test_semantic_true_mechanical_failure_is_blocked_with_exact_reason(self):
        raw = pd.DataFrame({
            "value": [10.5] * 30 + [None] * 10,
            "row_marker": list(range(40)),
        })
        report, cleaned, _, _ = self.run_fixture(raw)
        assessment = report.decision_summary["execution_routing"] \
            ["median_imputation_authorization_assessments"][0]
        self.assertEqual(assessment["semantic_eligibility"]["semantic_eligible"], "true")
        self.assertEqual(assessment["mechanical_eligibility"]["status"], "failed")
        self.assertEqual(
            assessment["mechanical_eligibility"]["reason_code"],
            "existing_imputation_contract_failed",
        )
        self.assertIn("requires LOW missingness", assessment["mechanical_eligibility"]["reason"])
        self.assertFalse(assessment["final_adjudication"]["authorized"])
        self.assertTrue(pd.isna(cleaned["value"].iloc[-1]))
        self.assertFalse(any(item.action_id == "impute_missing" for item in report.instructions))

    def test_false_and_unknown_semantics_block_mechanically_safe_values(self):
        cases = (
            ("nominal_code", "false", "nominal_numeric_code"),
            ("unknown", "unknown", "numeric_meaning_unknown"),
        )
        for meaning, expected_status, reason_code in cases:
            with self.subTest(meaning=meaning):
                raw = pd.DataFrame({
                    "value": [10.5, 20.5] * 20 + [None],
                    "row_marker": list(range(41)),
                })
                report, cleaned, _, _ = self.run_fixture(
                    raw, {"value": {"numeric_meaning": meaning}},
                )
                assessment = report.decision_summary["execution_routing"] \
                    ["median_imputation_authorization_assessments"][0]
                self.assertEqual(
                    assessment["semantic_eligibility"]["semantic_eligible"], expected_status,
                )
                self.assertEqual(assessment["semantic_eligibility"]["reason_code"], reason_code)
                self.assertEqual(assessment["mechanical_eligibility"]["status"], "not_evaluated")
                self.assertFalse(assessment["final_adjudication"]["authorized"])
                self.assertTrue(pd.isna(cleaned["value"].iloc[-1]))
                self.assertFalse(any(item.action_id == "impute_missing" for item in report.instructions))


if __name__ == "__main__":
    unittest.main()
