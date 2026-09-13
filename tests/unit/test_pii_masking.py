import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pipelines.pipeline1_cleaning.decisions.cleaning_decision_engine import (
    decide_cleaning,
)
from pipelines.pipeline1_cleaning.pipeline import run_pipeline1
from pipelines.pipeline1_cleaning.profiling.dataset_profiler import profile_dataset
from pipelines.pipeline1_cleaning.quality.issue_detector import detect_issues
from pipelines.pipeline1_cleaning.semantics.batched_schema import (
    BatchedColumnSemantic,
    BatchedDatasetContext,
    SemanticProfile,
)
from pipelines.pipeline1_cleaning.techniques.privacy_masking import mask_column


def _profile(column, semantic_type, role, function, description):
    return SemanticProfile(
        dataset_context=BatchedDatasetContext(
            row_entity="record",
            row_entity_confidence=0.9,
        ),
        columns=[
            BatchedColumnSemantic(
                column_name=column,
                semantic_type=semantic_type,
                semantic_role=role,
                value_function=function,
                numeric_meaning="not_applicable",
                concept_description=description,
                confidence=0.95,
                reasoning_summary="Fixture semantic evidence.",
            )
        ],
    )


class _NoLLMCalls:
    def generate_structured(self, prompt, response_model):
        raise AssertionError("PII Tier 1 routing must not call the LLM")


class PIIMaskingTests(unittest.TestCase):
    def test_mask_column_supports_hashing_and_full_redaction(self):
        source = pd.DataFrame({"value": ["same", None, "same", "different"]})
        hashed = mask_column(
            source.copy(),
            ["value"],
            {"strategy": "hash"},
        )
        redacted = mask_column(
            source.copy(),
            ["value"],
            {"strategy": "redact"},
        )

        self.assertEqual(hashed.loc[0, "value"], hashed.loc[2, "value"])
        self.assertNotEqual(hashed.loc[0, "value"], hashed.loc[3, "value"])
        self.assertTrue(hashed.loc[0, "value"].startswith("sha256:"))
        self.assertTrue(pd.isna(hashed.loc[1, "value"]))
        self.assertEqual(redacted["value"].dropna().unique().tolist(), ["[REDACTED]"])

    def test_direct_patterns_get_deterministic_subtypes_and_defaults(self):
        frame = pd.DataFrame({
            "email": ["a@example.com", "b@example.org", "c@example.net"],
            "phone": ["+1-202-555-0101", "+1-202-555-0102", "+1-202-555-0103"],
            "national_id": ["123-45-6789", "234-56-7890", "345-67-8901"],
        })
        issues = detect_issues(frame, profile_dataset(frame))
        pii = {
            issue.target_columns[0]: issue
            for issue in issues
            if issue.issue_type == "pii_candidate"
        }

        self.assertEqual(pii["email"].evidence["pii_subtype"], "email")
        self.assertEqual(
            pii["phone"].evidence["default_masking_strategy"],
            "hash",
        )
        self.assertEqual(
            pii["national_id"].evidence["default_masking_strategy"],
            "redact",
        )
        self.assertTrue(all(not issue.examples for issue in pii.values()))

    def test_phone_detection_rejects_numeric_lookalikes(self):
        frame = pd.DataFrame({
            'date': ['2024-01-02', '2024-02-03', '2024-03-04'],
            'ip': ['192.168.1.10', '10.0.0.12', '172.16.0.20'],
            'decimal': ['12345.6789', '23456.7890', '34567.8901'],
            'version': ['1.2.3.4567', '2.3.4.5678', '3.4.5.6789'],
        })
        issues = detect_issues(frame, profile_dataset(frame))

        self.assertFalse(any(
            issue.issue_type == 'pii_candidate'
            for issue in issues
        ))

    def test_person_name_uses_existing_semantic_result(self):
        frame = pd.DataFrame({"opaque": ["Alice Smith", "Bob Jones"]})
        semantic = _profile(
            "opaque",
            "text",
            "feature",
            "descriptive_attribute",
            "Employee full name.",
        ).columns[0].model_dump(mode="json")
        issues = detect_issues(
            frame,
            profile_dataset(frame),
            column_semantics={"opaque": semantic},
        )
        pii = [issue for issue in issues if issue.issue_type == "pii_candidate"]
        self.assertEqual(len(pii), 1)
        self.assertEqual(
            pii[0].evidence["detection_basis"],
            "existing_semantic_classification",
        )
        self.assertEqual(pii[0].evidence["pii_subtype"], "semantic_person_name")

    def test_pipeline_masks_and_keeps_privacy_out_of_quality_metrics(self):
        frame = pd.DataFrame({
            "contact": ["a@example.com", "b@example.org", "a@example.com"]
        })
        semantic_profile = _profile(
            "contact",
            "contact_information",
            "contact_field",
            "contact",
            "Email contact address.",
        )

        def semantic_analyzer(dataset_profile, dataset_evidence):
            return semantic_profile

        def decision_engine(issues, profile):
            return decide_cleaning(issues, profile, _NoLLMCalls())

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.csv"
            output = Path(directory) / "output"
            frame.to_csv(source, index=False)
            report = run_pipeline1(
                source,
                output,
                semantic_analyzer=semantic_analyzer,
                decision_engine=decision_engine,
            )
            cleaned = pd.read_csv(output / "cleaned_dataset.csv")

        self.assertEqual(report.status, "success")
        self.assertEqual(report.quality_summary["total_issues_detected"], 1)
        self.assertNotIn(
            "pii_candidate",
            report.quality_summary["issues_by_type"],
        )
        privacy = report.decision_summary["privacy_actions"]
        self.assertTrue(privacy["excluded_from_quality_fix_metrics"])
        self.assertEqual(privacy["reason"], "masked for privacy")
        self.assertEqual(privacy["masked_column_count"], 1)
        execution = next(
            item for item in report.executions if item.action_id == "mask_column"
        )
        self.assertEqual(execution.audit_metadata["reason"], "masked for privacy")
        self.assertEqual(execution.audit_metadata["metric_category"], "privacy")
        self.assertNotIn("before", execution.change_examples[0])
        self.assertEqual(cleaned.loc[0, "contact"], cleaned.loc[2, "contact"])
        self.assertNotEqual(cleaned.loc[0, "contact"], frame.loc[0, "contact"])
        self.assertEqual(
            report.before_profile["column_profiles"][0]["sample_values"],
            [],
        )


if __name__ == "__main__":
    unittest.main()
