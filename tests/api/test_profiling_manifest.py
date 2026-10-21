import json
import unittest

from api.profiling_manifest import build_profiling_manifest
from api.schemas import Pipeline1ResultResponse, ProfilingManifest


def _report(*, include_second_semantic=True):
    semantics = [{
        "column_name": "amount",
        "semantic_role": "measurement",
        "semantic_type": "numeric_measurement",
        "concept_description": "Observed transaction amount.",
        "confidence": 0.9,
        "reasoning_summary": "Numeric values represent transaction amounts.",
    }]
    if include_second_semantic:
        semantics.append({
            "column_name": "contact_email",
            "semantic_role": "unknown",
            "semantic_type": "email",
            "concept_description": "Customer contact email.",
            "confidence": 0.98,
            "reasoning_summary": "Values match an email pattern.",
        })
    return {
        "status": "success",
        "before_profile": {"column_profiles": [
            {
                "column_name": "amount", "physical_dtype": "float64",
                "non_null_count": 3, "missing": {"count": 1, "percentage": 25.0},
                "uniqueness": {"unique_count": 3}, "sample_values": [10.5, 20.0],
            },
            {
                "column_name": "contact_email", "physical_dtype": "object",
                "non_null_count": 4, "missing": {"count": 0, "percentage": 0.0},
                "uniqueness": {"unique_count": 4},
                "sample_values": ["alice@example.com", "bob@example.com"],
            },
        ]},
        "semantic_profile": {
            "columns": semantics,
            "telemetry": {"batch_failures": 0},
        },
        "decisions": [],
        "instructions": [],
        "executions": [],
    }


class ProfilingManifestTests(unittest.TestCase):
    def build(self, report=None, **kwargs):
        return build_profiling_manifest(
            report or _report(),
            dataset_name="transactions.csv",
            profiled_at="2026-10-12T09:30:00+00:00",
            privacy_clearance_passed=True,
            **kwargs,
        )

    def test_maps_observed_and_semantic_fields_without_inventing_roles(self):
        manifest = self.build()
        amount = manifest["columns"][0]
        self.assertEqual(manifest["total_columns"], 2)
        self.assertEqual(manifest["semantic_roles_detected"], ["measurement"])
        self.assertEqual(manifest["semantic_role_count"], 1)
        self.assertEqual(amount["physical_dtype"], "float64")
        self.assertEqual(amount["column_index"], 0)
        self.assertEqual(amount["semantic_role"], "measurement")
        self.assertEqual(amount["semantic_type"], "numeric_measurement")
        self.assertEqual(amount["non_null_count"], 3)
        self.assertEqual(amount["missing_count"], 1)
        self.assertEqual(amount["missing_percentage"], 25.0)
        self.assertEqual(amount["unique_count"], 3)
        self.assertEqual(amount["status"], "not_required")

    def test_missing_semantic_evidence_is_partial_and_remains_unresolved(self):
        manifest = self.build(_report(include_second_semantic=False))
        unresolved = manifest["columns"][1]
        self.assertEqual(manifest["profile_status"], "partial")
        self.assertIsNone(unresolved["semantic_role"])
        self.assertIsNone(unresolved["semantic_type"])
        self.assertIsNone(unresolved["decision_explanation"])
        self.assertEqual(unresolved["explanation_source"], "unavailable")

    def test_actions_join_to_execution_and_review_status_without_conflation(self):
        report = _report()
        report["decisions"] = [
            {
                "decision_id": "trim-1", "target_columns": ["amount"],
                "decision_type": "action", "proposed_action": "trim_whitespace",
                "rationale": "Whitespace was observed.", "evidence_summary": {"count": 2},
                "requires_review": False,
            },
            {
                "decision_id": "review-1", "target_columns": ["amount"],
                "decision_type": "flag_for_review", "proposed_action": "flag_rows",
                "rationale": "Range consequence is unknown.", "evidence_summary": {},
                "requires_review": True,
            },
        ]
        report["instructions"] = [{
            "source_decision_id": "trim-1", "action_id": "trim_whitespace",
        }]
        report["executions"] = [{
            "source_decision_id": "trim-1", "status": "success", "changed": True,
            "changed_cell_count": 2,
        }]
        manifest = self.build(report)
        amount = manifest["columns"][0]
        self.assertEqual([item["status"] for item in amount["actions"]], ["requires_review", "applied"])
        self.assertEqual(amount["status"], "requires_review")
        self.assertEqual(amount["primary_action"]["action_id"], "flag_rows")
        self.assertEqual(manifest["columns_requiring_cleaning"], 1)

    def test_applied_action_is_not_counted_as_still_requiring_cleaning(self):
        report = _report()
        report["decisions"] = [{
            "decision_id": "trim-1", "target_columns": ["amount"],
            "decision_type": "action", "proposed_action": "trim_whitespace",
            "rationale": "Whitespace was observed.", "evidence_summary": {},
            "requires_review": False,
        }]
        report["instructions"] = [{"source_decision_id": "trim-1", "action_id": "trim_whitespace"}]
        report["executions"] = [{
            "source_decision_id": "trim-1", "status": "success", "changed": True,
            "changed_cell_count": 1,
        }]
        manifest = self.build(report)
        self.assertEqual(manifest["columns"][0]["status"], "applied")
        self.assertEqual(manifest["columns_requiring_cleaning"], 0)

    def test_proposed_action_is_not_reported_as_applied(self):
        report = _report()
        report["decisions"] = [{
            "decision_id": "pending-1", "target_columns": ["amount"],
            "decision_type": "action", "proposed_action": "coerce_dtype",
            "rationale": "A numeric conversion was proposed.", "evidence_summary": {},
            "requires_review": False,
        }]
        manifest = self.build(report)
        self.assertEqual(manifest["columns"][0]["status"], "proposed")

    def test_observed_examples_are_redacted_or_withheld_by_privacy_policy(self):
        redacted = self.build(pii_columns=["contact_email"])
        email = redacted["columns"][1]
        self.assertTrue(email["privacy_restricted"])
        self.assertEqual(email["observed_examples"], ["[REDACTED]"])

        blocked = build_profiling_manifest(
            _report(), dataset_name="transactions.csv", profiled_at=None,
            privacy_clearance_passed=False, pii_columns=[],
        )
        self.assertTrue(all(column["privacy_restricted"] for column in blocked["columns"]))
        self.assertTrue(all(column["observed_examples"] == [] for column in blocked["columns"]))

    def test_manifest_schema_serializes_backward_compatible_payload(self):
        manifest = ProfilingManifest.model_validate(self.build())
        payload = manifest.model_dump(mode="json")
        json.dumps(payload, allow_nan=False)
        self.assertEqual(payload["dataset_name"], "transactions.csv")

        legacy = Pipeline1ResultResponse.model_validate({
            "run_id": "legacy-run", "state": "complete",
            "pipeline1_internal_run_id": "pipeline-run", "pipeline1_report": {},
            "prepared_dataset": {"rows": 0, "columns": 0, "target_columns": []},
            "privacy_clearance_passed": True,
            "prediction_eligibility": {
                "eligible": False, "reasons": [],
                "cleaned_dataset_sha256_verified": False,
            },
        })
        self.assertIsNone(legacy.profiling_manifest)


if __name__ == "__main__":
    unittest.main()
