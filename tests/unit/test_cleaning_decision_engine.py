import unittest

import pandas as pd

from pipelines.pipeline1_cleaning.decisions.cleaning_decision_engine import (
    DecisionConfig, Tier2DecisionBatch, Tier2DecisionPayload,
    Tier2MissingDecisionBatch, _missing_value_readiness, _missing_values_prompt,
    _normalize_missing_value_decision, _validate_coverage,
    _validate_missing_value_alignment,
    decide_cleaning,
)
from pipelines.pipeline1_cleaning.decisions.decision_schema import CleaningDecision
from pipelines.pipeline1_cleaning.quality.quality_report import DetectedIssue
from pipelines.pipeline1_cleaning.semantics.batched_schema import (
    BatchedColumnSemantic, BatchedDatasetContext, SemanticProfile,
)


def issue(kind, name="value", evidence=None):
    return DetectedIssue(
        issue_id=f"issue-{kind}-{name}", issue_type=kind,
        target_columns=[] if name is None else [name], affected_count=1,
        affected_percentage=10.0, evidence=evidence or {}, examples=["x"],
        status="candidate", risk_hint="context_dependent",
    )


def profile(function="unknown"):
    numeric_meaning = {
        "measurement": "magnitude",
        "entity_reference": "identifier",
        "unknown": "unknown",
    }.get(function, "not_applicable")
    semantic_type = {
        "measurement": "numeric_measurement",
        "entity_reference": "identifier",
        "classification": "categorical",
        "temporal": "time",
    }.get(function, "unknown")
    return SemanticProfile(
        dataset_context=BatchedDatasetContext(row_entity="record", row_entity_confidence=0.8),
        columns=[BatchedColumnSemantic(
            column_name="value", semantic_type=semantic_type, value_function=function,
            numeric_meaning=numeric_meaning,
            concept_description="Test values.", semantic_role="unknown",
            confidence=0.8, reasoning_summary="Mock semantics.",
        )],
    )


class MockLLM:
    last_response_metadata = {}

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0
        self.prompts = []

    def generate_structured(self, prompt, response_model):
        self.calls += 1
        self.prompts.append(prompt)
        if self.fail:
            raise RuntimeError("unavailable")
        marker = '\"issue_id\":\"'
        issue_ids = []
        remaining = prompt
        while marker in remaining:
            remaining = remaining.split(marker, 1)[1]
            issue_id, remaining = remaining.split('\"', 1)
            if issue_id not in issue_ids:
                issue_ids.append(issue_id)
        decisions = []
        for issue_id in issue_ids:
            if "MISSING-VALUE DECISION RUBRIC" in prompt:
                decisions.append(dict(
                    issue_id=issue_id, decision_type="action",
                    proposed_action="impute_missing", parameters={"strategy": "median"},
                    recommendation_basis="semantically_appropriate_imputation",
                    confidence=0.99, rationale="Mock missing-value recommendation.",
                ))
                continue
            if "mixed_numeric" in issue_id:
                action = "coerce_dtype"
            elif "date" in issue_id:
                action = "parse_dates"
            else:
                action = "impute_missing"
            decisions.append(dict(
                issue_id=issue_id, decision_type="action",
                proposed_action=action, confidence=0.99, rationale="Mock proposal.",
            ))
        return {"decisions": decisions}


def median_action(issue_id):
    return {
        "issue_id": issue_id, "decision_type": "action",
        "proposed_action": "impute_missing", "parameters": {"strategy": "median"},
        "recommendation_basis": "semantically_appropriate_imputation",
        "confidence": 0.9, "rationale": "Context supports median imputation.",
    }


def unsupported_no_action(issue_id):
    return {
        "issue_id": issue_id, "decision_type": "no_action",
        "proposed_action": "no_op", "parameters": {},
        "recommendation_basis": "affirmative_preservation_evidence",
        "preservation_evidence_keys": ["missingness_is_meaningful"],
        "confidence": 0.9, "rationale": "Missingness should be preserved.",
    }


class ScriptedLLM:
    last_response_metadata = {}

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_structured(self, prompt, response_model):
        marker = '\"issue_id\":\"'
        issue_ids = []
        remaining = prompt
        while marker in remaining:
            remaining = remaining.split(marker, 1)[1]
            issue_id, remaining = remaining.split('\"', 1)
            if issue_id not in issue_ids:
                issue_ids.append(issue_id)
        self.calls.append(issue_ids)
        response = self.responses[len(self.calls) - 1]
        if isinstance(response, Exception):
            raise response
        return response(issue_ids) if callable(response) else response


def missing_measurement_fixture(count=4):
    names = [f"metric_{index}" for index in range(count)]
    detected = [
        issue("missing_values", name, evidence={"non_null_count": 40})
        for name in names
    ]
    semantics = SemanticProfile(
        dataset_context=BatchedDatasetContext(
            row_entity="record", row_entity_confidence=0.8,
        ),
        columns=[BatchedColumnSemantic(
            column_name=name, semantic_type="numeric_measurement",
            value_function="measurement", numeric_meaning="magnitude",
            concept_description="A measured quantity.", semantic_role="unknown",
            confidence=0.9, reasoning_summary="Quantitative evidence.",
        ) for name in names],
    )
    values = pd.DataFrame({
        name: [10.5, 20.5] * 20 + [None] for name in names
    })
    return detected, semantics, values


class CleaningDecisionEngineTests(unittest.TestCase):
    def decide(self, detected, semantics=None, llm=None):
        return decide_cleaning(detected, semantics or profile(), llm or MockLLM())

    def test_whitespace_is_deterministic_trim(self):
        decisions, summary, _ = self.decide([issue("whitespace_issue")])
        self.assertEqual(decisions[0].proposed_action, "trim_whitespace")
        self.assertEqual(summary["tier1_decisions"], 1)

    def test_rare_category_is_no_action(self):
        decisions, _, _ = self.decide([issue("rare_category_observation")])
        self.assertEqual(decisions[0].decision_type, "no_action")

    def test_outlier_and_duplicate_are_flagged(self):
        detected = [issue("statistical_outlier_candidate"), issue("exact_duplicate_rows", None)]
        decisions, _, _ = self.decide(detected)
        self.assertTrue(all(item.decision_type == "flag_for_review" for item in decisions))

    def test_missing_entity_reference_is_not_imputed(self):
        decisions, _, _ = self.decide([issue("missing_values")], profile("entity_reference"))
        self.assertEqual(decisions[0].proposed_action, "flag_rows")

    def test_unknown_semantics_preserves_data(self):
        decisions, _, _ = self.decide([issue("missing_values")], profile("unknown"))
        self.assertEqual(decisions[0].decision_type, "insufficient_evidence")
        self.assertEqual(decisions[0].proposed_action, "no_op")

    def test_measurement_missing_and_numeric_mixed_route_to_batched_tier2(self):
        detected = [issue("missing_values"), issue("mixed_numeric_representation_candidate")]
        llm = MockLLM()
        values = pd.DataFrame({"value": [10.5, 20.5] * 20 + [None]})
        decisions, summary, _ = decide_cleaning(
            detected, profile("measurement"), llm, dataframe=values,
        )
        self.assertEqual(llm.calls, 2)
        self.assertEqual(summary["tier2_batch_kinds"], ["general", "missing_values"])
        self.assertEqual(summary["tier2_issue_count"], 2)
        self.assertEqual(
            {item.proposed_action for item in decisions},
            {"impute_missing", "coerce_dtype"},
        )
        self.assertTrue(all(item.requires_review for item in decisions))

    def test_temporal_date_candidate_routes_to_tier2(self):
        llm = MockLLM()
        decisions, _, _ = self.decide(
            [issue("inconsistent_date_format_candidate")], profile("temporal"), llm
        )
        self.assertEqual(decisions[0].proposed_action, "parse_dates")
        self.assertEqual(decisions[0].decision_source, "llm_tier2")

    def test_lineage_is_preserved(self):
        detected = issue("rare_category_observation")
        decisions, _, _ = self.decide([detected])
        self.assertEqual(decisions[0].source_issue_ids, [detected.issue_id])

    def test_tier2_batches_multiple_issues(self):
        detected = [issue("case_variant_candidate", f"value{i}") for i in range(9)]
        semantics = SemanticProfile(
            dataset_context=BatchedDatasetContext(row_entity="record", row_entity_confidence=0.8),
            columns=[],
        )
        llm = MockLLM()
        decisions, summary, _ = decide_cleaning(detected, semantics, llm)
        self.assertEqual(summary["tier2_batch_sizes"], [4, 4, 1])
        self.assertEqual(summary["intended_llm_decision_calls"], 3)
        self.assertEqual(summary["llm_decision_calls"], 3)
        self.assertTrue(all(item.source_issue_ids for item in decisions))
        self.assertTrue(all(isinstance(item.evidence_summary, dict) for item in decisions))
        self.assertTrue(all(item.decision_source == "llm_tier2" for item in decisions))

    def test_missing_value_prompt_contains_redacted_readiness_and_rubric(self):
        detected = issue(
            "missing_values", evidence={"null_detection": "pandas.isna", "non_null_count": 40},
        )
        detected.examples = [987654]
        values = pd.DataFrame({"value": [10.25, 20.25] * 20 + [None]})
        prompt = _missing_values_prompt([detected], profile("measurement"), values)
        readiness = _missing_value_readiness(detected, profile("measurement"), values)

        for field in (
            "missing_count", "missing_percentage", "missingness_band", "observed_count",
            "physical_dtype", "semantic_type", "value_function", "numeric_meaning",
            "semantic_confidence", "concept_description",
            "identifier_or_entity_reference_protected", "median_semantic_eligibility",
            "median_mechanical_preflight",
        ):
            self.assertIn(field, prompt)
            self.assertIn(field, readiness)
        self.assertIn('"numeric_meaning":"magnitude"', prompt)
        self.assertIn('"status":"true"', prompt)
        self.assertIn('"status":"pass"', prompt)
        self.assertNotIn("replacement_value", prompt)
        self.assertNotIn("15.25", prompt)
        self.assertNotIn("987654", prompt)
        self.assertIn("RECOMMENDATION ONLY", prompt)
        self.assertIn("Low missingness alone is not a preservation reason", prompt)
        self.assertIn("Uncertainty is not a reason", prompt)
        self.assertIn("affirmative_preservation_evidence", prompt)

    def test_missing_value_batches_are_homogeneous_and_bounded(self):
        detected = (
            [issue("missing_values", f"missing{i}") for i in range(9)]
            + [issue("case_variant_candidate", f"case{i}") for i in range(5)]
        )
        semantics = SemanticProfile(
            dataset_context=BatchedDatasetContext(row_entity="record", row_entity_confidence=0.8),
            columns=[BatchedColumnSemantic(
                column_name=name, semantic_type="numeric_measurement",
                value_function="measurement", numeric_meaning="magnitude",
                concept_description="A measured value.", confidence=0.9,
                reasoning_summary="Fixture semantics.",
            ) for name in [f"missing{i}" for i in range(9)]],
        )
        llm = MockLLM()
        decisions, summary, _ = decide_cleaning(detected, semantics, llm)
        self.assertEqual(len(decisions), 14)
        self.assertEqual(summary["tier2_batch_sizes"], [4, 1, 4, 4, 1])
        self.assertEqual(summary["tier2_batch_kinds"], [
            "general", "general", "missing_values", "missing_values", "missing_values",
        ])
        for kind, prompt in zip(summary["tier2_batch_kinds"], llm.prompts):
            if kind == "missing_values":
                self.assertIn("MISSING-VALUE DECISION RUBRIC", prompt)
                self.assertNotIn("case_variant_candidate", prompt)
            else:
                self.assertNotIn("MISSING-VALUE DECISION RUBRIC", prompt)
                self.assertNotIn('"issue_type":"missing_values"', prompt)
        self.assertLessEqual(max(summary["tier2_batch_sizes"]), 4)

    def test_missing_value_readiness_reports_uncertainty_and_mechanical_failure(self):
        detected = issue("missing_values", evidence={"non_null_count": 30})
        values = pd.DataFrame({"value": [10.5] * 30 + [None] * 10})
        uncertain = profile("unknown")
        readiness = _missing_value_readiness(detected, uncertain, values)
        self.assertEqual(readiness["median_semantic_eligibility"]["status"], "unknown")
        self.assertEqual(
            readiness["median_semantic_eligibility"]["reason_code"], "numeric_meaning_unknown",
        )
        self.assertEqual(readiness["median_mechanical_preflight"]["status"], "fail")
        self.assertEqual(
            readiness["median_mechanical_preflight"]["reason_code"],
            "existing_imputation_contract_failed",
        )
        self.assertIn("requires LOW missingness", readiness["median_mechanical_preflight"]["reason"])

    def test_no_action_rubric_requires_affirmative_preservation_evidence(self):
        detected = issue("missing_values", evidence={
            "non_null_count": 40,
            "missingness_is_meaningful": True,
            "missingness_justification": "The source contract defines absence as not applicable.",
        })
        values = pd.DataFrame({"value": [10.5, 20.5] * 20 + [None]})
        prompt = _missing_values_prompt([detected], profile("measurement"), values)
        readiness = _missing_value_readiness(detected, profile("measurement"), values)
        self.assertEqual(readiness["affirmative_preservation_evidence"], {
            "missingness_is_meaningful": True,
            "missingness_justification": "The source contract defines absence as not applicable.",
        })
        self.assertIn("Use only when affirmative_preservation_evidence", prompt)
        self.assertIn("Cite that concrete evidence", prompt)

    def test_temporal_and_identifier_readiness_cannot_support_median(self):
        values = pd.DataFrame({"value": ["12:00:00"] * 40 + [None]})
        temporal = SemanticProfile(
            dataset_context=profile().dataset_context,
            columns=[BatchedColumnSemantic(
                column_name="value", semantic_type="time", value_function="temporal",
                numeric_meaning="not_applicable", concept_description="Time of an event.",
                confidence=0.9, reasoning_summary="Temporal evidence.",
            )],
        )
        temporal_readiness = _missing_value_readiness(issue("missing_values"), temporal, values)
        self.assertEqual(temporal_readiness["median_semantic_eligibility"]["status"], "false")
        self.assertEqual(temporal_readiness["median_mechanical_preflight"]["status"], "fail")
        identifier = _missing_value_readiness(
            issue("missing_values"), profile("entity_reference"),
            pd.DataFrame({"value": [1001.0] * 40 + [None]}),
        )
        self.assertTrue(identifier["identifier_or_entity_reference_protected"])
        self.assertEqual(identifier["median_semantic_eligibility"]["status"], "false")

    def test_exact_issue_coverage_validation_is_unchanged(self):
        one = Tier2DecisionPayload(
            issue_id="issue-1", decision_type="insufficient_evidence",
            proposed_action="no_op", confidence=0.8, rationale="Evidence is inadequate.",
        )
        with self.assertRaisesRegex(ValueError, "exactly cover"):
            _validate_coverage(["issue-1", "issue-2"], [one])

    def test_missing_specific_schema_requires_complete_median_action(self):
        valid = {
            "issue_id": "issue-1", "decision_type": "action",
            "proposed_action": "impute_missing", "parameters": {"strategy": "median"},
            "recommendation_basis": "semantically_appropriate_imputation",
            "confidence": 0.9, "rationale": "Context supports filling.",
        }
        parsed = Tier2MissingDecisionBatch.model_validate({"decisions": [valid]})
        normalized = _normalize_missing_value_decision(parsed.decisions[0])
        self.assertEqual(normalized.parameters, {"strategy": "median"})
        for change in (
            {"parameters": {}},
            {"parameters": {"strategy": "mode"}},
            {"proposed_action": "no_op"},
            {"recommendation_basis": "affirmative_preservation_evidence"},
        ):
            with self.subTest(change=change), self.assertRaises(Exception):
                Tier2MissingDecisionBatch.model_validate({"decisions": [{**valid, **change}]})

    def test_missing_specific_schema_requires_typed_non_action_bases(self):
        base = {"issue_id": "issue-1", "parameters": {}, "confidence": 0.9,
                "rationale": "Evidence supports this outcome."}
        valid = (
            {**base, "decision_type": "no_action", "proposed_action": "no_op",
             "recommendation_basis": "affirmative_preservation_evidence",
             "preservation_evidence_keys": ["missingness_is_meaningful"]},
            {**base, "decision_type": "insufficient_evidence", "proposed_action": "no_op",
             "recommendation_basis": "insufficient_or_contradictory_evidence"},
            {**base, "decision_type": "flag_for_review", "proposed_action": "flag_rows",
             "review_basis": "high_missingness_requires_review"},
        )
        for payload in valid:
            Tier2MissingDecisionBatch.model_validate({"decisions": [payload]})
        with self.assertRaises(Exception):
            Tier2MissingDecisionBatch.model_validate({"decisions": [{
                **valid[0], "preservation_evidence_keys": [],
            }]})

    def test_readiness_alignment_rejects_contradictions_without_choosing_outcome(self):
        detected = issue("missing_values", evidence={"non_null_count": 40})
        safe_values = pd.DataFrame({"value": [10.5, 20.5] * 20 + [None]})
        safe = _missing_value_readiness(detected, profile("measurement"), safe_values)
        action = Tier2MissingDecisionBatch.model_validate({"decisions": [{
            "issue_id": detected.issue_id, "decision_type": "action",
            "proposed_action": "impute_missing", "parameters": {"strategy": "median"},
            "recommendation_basis": "semantically_appropriate_imputation",
            "confidence": 0.9, "rationale": "Filling is contextually appropriate.",
        }]}).decisions[0]
        _validate_missing_value_alignment(action, safe)

        source_shape = pd.DataFrame({"value": list(range(1000, 1040)) + [None]})
        unknown = _missing_value_readiness(detected, profile("measurement"), source_shape)
        self.assertEqual(unknown["median_semantic_eligibility"]["status"], "unknown")
        with self.assertRaisesRegex(ValueError, "semantic eligibility is not true"):
            _validate_missing_value_alignment(action, unknown)

        mechanical_fail = _missing_value_readiness(
            detected, profile("measurement"),
            pd.DataFrame({"value": [10.5] * 30 + [None] * 10}),
        )
        with self.assertRaisesRegex(ValueError, "mechanical preflight did not pass"):
            _validate_missing_value_alignment(action, mechanical_fail)

    def test_no_action_and_review_bases_must_match_supplied_readiness(self):
        base = {"issue_id": "issue-1", "parameters": {}, "confidence": 0.9,
                "rationale": "Structured basis under test."}
        no_action = Tier2MissingDecisionBatch.model_validate({"decisions": [{
            **base, "decision_type": "no_action", "proposed_action": "no_op",
            "recommendation_basis": "affirmative_preservation_evidence",
            "preservation_evidence_keys": ["missingness_is_meaningful"],
        }]}).decisions[0]
        values = pd.DataFrame({"value": [10.5, 20.5] * 20 + [None]})
        absent = _missing_value_readiness(issue("missing_values"), profile("measurement"), values)
        with self.assertRaisesRegex(ValueError, "requires supplied affirmative"):
            _validate_missing_value_alignment(no_action, absent)
        supported_issue = issue("missing_values", evidence={
            "missingness_is_meaningful": True,
            "missingness_justification": "The source defines absence as meaningful.",
        })
        supported = _missing_value_readiness(supported_issue, profile("measurement"), values)
        _validate_missing_value_alignment(no_action, supported)

        flag_cases = (
            ("protected_entity_reference", profile("entity_reference"),
             pd.DataFrame({"value": [1001.0] * 40 + [None]})),
            ("temporal_value_not_safely_synthesizable", profile("temporal"),
             pd.DataFrame({"value": ["12:00:00"] * 40 + [None]})),
            ("high_missingness_requires_review", profile("classification"),
             pd.DataFrame({"value": ["A"] * 4 + [None] * 6})),
        )
        for review_basis, semantics, frame in flag_cases:
            with self.subTest(review_basis=review_basis):
                flag = Tier2MissingDecisionBatch.model_validate({"decisions": [{
                    **base, "decision_type": "flag_for_review", "proposed_action": "flag_rows",
                    "review_basis": review_basis,
                }]}).decisions[0]
                readiness = _missing_value_readiness(issue("missing_values"), semantics, frame)
                _validate_missing_value_alignment(flag, readiness)
                wrong = dict(readiness)
                wrong.update({"identifier_or_entity_reference_protected": False,
                              "semantic_type": "categorical", "value_function": "classification",
                              "missingness_band": "low"})
                with self.assertRaisesRegex(ValueError, "unsupported by readiness"):
                    _validate_missing_value_alignment(flag, wrong)

    def test_invalid_missing_response_uses_existing_retry_then_accepts(self):
        detected = issue("missing_values", evidence={"non_null_count": 40})
        valid = {
            "issue_id": detected.issue_id, "decision_type": "action",
            "proposed_action": "impute_missing", "parameters": {"strategy": "median"},
            "recommendation_basis": "semantically_appropriate_imputation",
            "confidence": 0.9, "rationale": "Context supports median imputation.",
        }

        class SequenceLLM:
            last_response_metadata = {}

            def __init__(self, responses):
                self.responses = list(responses)
                self.calls = 0

            def generate_structured(self, prompt, response_model):
                response = self.responses[self.calls]
                self.calls += 1
                return response

        bad = {"decisions": [{**valid, "parameters": {}}]}
        good = {"decisions": [valid]}
        values = pd.DataFrame({"value": [10.5, 20.5] * 20 + [None]})
        llm = SequenceLLM([bad, good])
        decisions, summary, warnings = decide_cleaning(
            [detected], profile("measurement"), llm, dataframe=values,
        )
        self.assertEqual(llm.calls, 2)
        self.assertEqual(summary["retries"], 1)
        self.assertEqual(summary["batch_failures"], 0)
        self.assertEqual(decisions[0].proposed_action, "impute_missing")
        self.assertEqual(decisions[0].parameters, {"strategy": "median"})
        self.assertFalse(warnings)

        llm = SequenceLLM([bad, bad])
        decisions, summary, warnings = decide_cleaning(
            [detected], profile("measurement"), llm, dataframe=values,
        )
        self.assertEqual(llm.calls, 2)
        self.assertEqual(summary["batch_failures"], 1)
        self.assertEqual(decisions[0].decision_type, "insufficient_evidence")
        self.assertEqual(decisions[0].decision_source, "fallback")
        self.assertTrue(warnings)

    def test_salvage_three_valid_siblings_and_retry_only_one_invalid(self):
        detected, semantics, values = missing_measurement_fixture(4)
        ids = [item.issue_id for item in detected]
        llm = ScriptedLLM([
            {"decisions": [
                median_action(ids[0]), median_action(ids[1]),
                median_action(ids[2]), unsupported_no_action(ids[3]),
            ]},
            {"decisions": [unsupported_no_action(ids[3])]},
        ])
        decisions, summary, _ = decide_cleaning(
            detected, semantics, llm, dataframe=values,
        )
        self.assertEqual(llm.calls, [ids, [ids[3]]])
        self.assertEqual(sum(d.decision_source == "llm_tier2" for d in decisions), 3)
        self.assertEqual(decisions[3].decision_source, "fallback")
        self.assertEqual(summary["tier2_recovery_audit"][0]["accepted_on_first_response"], ids[:3])

    def test_one_valid_sibling_survives_three_invalid_decisions(self):
        detected, semantics, values = missing_measurement_fixture(4)
        ids = [item.issue_id for item in detected]
        llm = ScriptedLLM([
            {"decisions": [median_action(ids[0])] + [
                unsupported_no_action(issue_id) for issue_id in ids[1:]
            ]},
            lambda subset: {"decisions": [unsupported_no_action(item) for item in subset]},
            lambda subset: {"decisions": [unsupported_no_action(item) for item in subset]},
        ])
        decisions, _, _ = decide_cleaning(detected, semantics, llm, dataframe=values)
        self.assertEqual(llm.calls[0], ids)
        self.assertNotIn(ids[0], llm.calls[1] + llm.calls[2])
        self.assertEqual(decisions[0].decision_source, "llm_tier2")
        self.assertTrue(all(d.decision_source == "fallback" for d in decisions[1:]))

    def test_retry_repairs_only_invalid_issue(self):
        detected, semantics, values = missing_measurement_fixture(2)
        ids = [item.issue_id for item in detected]
        llm = ScriptedLLM([
            {"decisions": [median_action(ids[0]), unsupported_no_action(ids[1])]},
            {"decisions": [median_action(ids[1])]},
        ])
        decisions, summary, _ = decide_cleaning(detected, semantics, llm, dataframe=values)
        self.assertEqual(llm.calls, [ids, [ids[1]]])
        self.assertTrue(all(d.decision_source == "llm_tier2" for d in decisions))
        self.assertEqual(summary["tier2_recovery_audit"][0]["recovered_on_retry"], [ids[1]])

    def test_retry_invalid_falls_back_only_that_issue(self):
        detected, semantics, values = missing_measurement_fixture(2)
        ids = [item.issue_id for item in detected]
        llm = ScriptedLLM([
            {"decisions": [median_action(ids[0]), unsupported_no_action(ids[1])]},
            {"decisions": [unsupported_no_action(ids[1])]},
        ])
        decisions, summary, _ = decide_cleaning(detected, semantics, llm, dataframe=values)
        self.assertEqual(decisions[0].decision_source, "llm_tier2")
        self.assertEqual(decisions[1].decision_source, "fallback")
        self.assertEqual(summary["fallback_issue_count"], 1)

    def test_missing_issue_id_retries_missing_only_and_retains_siblings(self):
        detected, semantics, values = missing_measurement_fixture(3)
        ids = [item.issue_id for item in detected]
        llm = ScriptedLLM([
            {"decisions": [median_action(ids[0]), median_action(ids[2])]},
            {"decisions": [median_action(ids[1])]},
        ])
        decisions, summary, _ = decide_cleaning(detected, semantics, llm, dataframe=values)
        self.assertEqual(llm.calls, [ids, [ids[1]]])
        self.assertTrue(all(d.decision_source == "llm_tier2" for d in decisions))
        coverage = summary["tier2_recovery_audit"][0]["calls"][0]["coverage"]
        self.assertEqual(coverage["missing_issue_ids"], [ids[1]])

    def test_unexpected_issue_id_is_never_accepted(self):
        detected, semantics, values = missing_measurement_fixture(2)
        ids = [item.issue_id for item in detected]
        llm = ScriptedLLM([{"decisions": [
            median_action(ids[0]), median_action(ids[1]), median_action("unexpected"),
        ]}])
        decisions, summary, _ = decide_cleaning(detected, semantics, llm, dataframe=values)
        self.assertEqual(len(decisions), 2)
        self.assertTrue(all(d.source_issue_ids[0] in ids for d in decisions))
        coverage = summary["tier2_recovery_audit"][0]["calls"][0]["coverage"]
        self.assertEqual(coverage["unexpected_issue_ids"], ["unexpected"])

    def test_duplicate_issue_id_retries_affected_issue_only(self):
        detected, semantics, values = missing_measurement_fixture(2)
        ids = [item.issue_id for item in detected]
        llm = ScriptedLLM([
            {"decisions": [
                median_action(ids[0]), median_action(ids[0]), median_action(ids[1]),
            ]},
            {"decisions": [median_action(ids[0])]},
        ])
        decisions, summary, _ = decide_cleaning(detected, semantics, llm, dataframe=values)
        self.assertEqual(llm.calls, [ids, [ids[0]]])
        self.assertTrue(all(d.decision_source == "llm_tier2" for d in decisions))
        coverage = summary["tier2_recovery_audit"][0]["calls"][0]["coverage"]
        self.assertEqual(coverage["duplicate_issue_ids"], [ids[0]])

    def test_initial_timeout_splits_unresolved_batch(self):
        detected, semantics, values = missing_measurement_fixture(4)
        ids = [item.issue_id for item in detected]
        llm = ScriptedLLM([
            TimeoutError("provider timed out"),
            lambda subset: {"decisions": [median_action(item) for item in subset]},
            lambda subset: {"decisions": [median_action(item) for item in subset]},
        ])
        decisions, summary, _ = decide_cleaning(detected, semantics, llm, dataframe=values)
        self.assertEqual(llm.calls, [ids, ids[:2], ids[2:]])
        self.assertTrue(all(d.decision_source == "llm_tier2" for d in decisions))
        self.assertEqual(summary["tier2_recovery_audit"][0]["calls"][0]["provider_status"], "timeout")

    def test_individual_timeout_falls_back_only_timed_out_issue(self):
        detected, semantics, values = missing_measurement_fixture(2)
        ids = [item.issue_id for item in detected]
        llm = ScriptedLLM([
            TimeoutError("provider timed out"), TimeoutError("provider timed out"),
            lambda subset: {"decisions": [median_action(item) for item in subset]},
        ])
        decisions, summary, _ = decide_cleaning(detected, semantics, llm, dataframe=values)
        self.assertEqual(llm.calls, [ids, [ids[0]], [ids[1]]])
        self.assertEqual(decisions[0].decision_source, "fallback")
        self.assertEqual(decisions[1].decision_source, "llm_tier2")
        self.assertEqual(summary["fallback_issue_count"], 1)

    def test_previously_accepted_decision_is_never_regenerated(self):
        detected, semantics, values = missing_measurement_fixture(4)
        ids = [item.issue_id for item in detected]
        llm = ScriptedLLM([
            {"decisions": [median_action(ids[0])] + [
                unsupported_no_action(issue_id) for issue_id in ids[1:]
            ]},
            lambda subset: {"decisions": [median_action(item) for item in subset]},
            lambda subset: {"decisions": [median_action(item) for item in subset]},
        ])
        decide_cleaning(detected, semantics, llm, dataframe=values)
        self.assertTrue(all(ids[0] not in retry_call for retry_call in llm.calls[1:]))

    def test_compact_payload_expands_to_complete_lineage(self):
        detected = issue("case_variant_candidate", evidence={"probe": "casefold"})
        decisions, _, _ = self.decide([detected], profile("classification"), MockLLM())
        decision = decisions[0]
        self.assertEqual(decision.decision_id, f"decision-{detected.issue_id}")
        self.assertEqual(decision.source_issue_ids, [detected.issue_id])
        self.assertEqual(decision.target_columns, detected.target_columns)
        self.assertEqual(decision.evidence_summary, detected.evidence)
        self.assertEqual(decision.decision_source, "llm_tier2")

    def test_compact_payload_rejects_unsupported_action(self):
        with self.assertRaises(ValueError):
            Tier2DecisionBatch.model_validate({"decisions": [{
                "issue_id": "issue-1", "decision_type": "action",
                "proposed_action": "execute_arbitrary_code", "parameters": {},
                "confidence": 0.5, "rationale": "Unsupported.",
            }]})

    def test_llm_failure_retries_then_falls_back(self):
        llm = MockLLM(fail=True)
        decisions, summary, warnings = self.decide(
            [issue("case_variant_candidate")], profile("classification"), llm
        )
        self.assertEqual(summary["retries"], 1)
        self.assertEqual(summary["batch_failures"], 1)
        self.assertEqual(decisions[0].decision_type, "insufficient_evidence")
        self.assertTrue(warnings)


if __name__ == "__main__":
    unittest.main()
