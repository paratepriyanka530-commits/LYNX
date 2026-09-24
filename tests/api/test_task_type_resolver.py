import unittest
from uuid import uuid4

import numpy as np
import pandas as pd

from api.task_type_resolver import TaskTypeResolution, resolve_task_type


def _report(column, **overrides):
    semantic = {
        "column_name": column,
        "semantic_type": "unknown",
        "semantic_role": "target_candidate",
        "value_function": "unknown",
        "numeric_meaning": "unknown",
        "concept_description": "selected target",
        "confidence": 0.95,
        "reasoning_summary": "test evidence",
    }
    semantic.update(overrides)
    return {"semantic_profile": {"columns": [semantic], "telemetry": {}}}


class TaskTypeResolverTests(unittest.TestCase):
    def assert_resolution(self, result, status, task_type):
        self.assertIsInstance(result, TaskTypeResolution)
        self.assertEqual(result.status, status)
        self.assertEqual(result.task_type, task_type)
        self.assertTrue(result.reason)
        self.assertEqual(result.to_dict()["status"], status)

    def test_boolean_target_is_classification(self):
        result = resolve_task_type(pd.DataFrame({"target": [True, False] * 4}), "target")
        self.assert_resolution(result, "resolved", "classification")

    def test_binary_string_labels_are_classification(self):
        frame = pd.DataFrame({"target": ["yes", "no", "yes", "no", "yes", "no"]})
        self.assert_resolution(resolve_task_type(frame, "target"), "resolved", "classification")

    def test_multiclass_string_labels_are_classification(self):
        frame = pd.DataFrame({"target": ["red", "blue", "green"] * 4})
        self.assert_resolution(resolve_task_type(frame, "target"), "resolved", "classification")

    def test_continuous_float_measurement_is_regression(self):
        values = np.linspace(0.15, 9.85, 30)
        self.assert_resolution(
            resolve_task_type(pd.DataFrame({"target": values}), "target"),
            "resolved", "regression",
        )

    def test_continuous_integer_measurement_with_magnitude_semantics_is_regression(self):
        frame = pd.DataFrame({"duration": list(range(10, 110, 5))})
        report = _report(
            "duration", semantic_type="numeric_measurement",
            semantic_role="measurement", value_function="measurement",
            numeric_meaning="magnitude",
        )
        self.assert_resolution(
            resolve_task_type(frame, "duration", report), "resolved", "regression"
        )

    def test_numeric_binary_with_classification_semantics(self):
        frame = pd.DataFrame({"churn": [0, 1] * 10})
        report = _report(
            "churn", semantic_type="boolean", value_function="classification",
            numeric_meaning="nominal_code",
        )
        self.assert_resolution(
            resolve_task_type(frame, "churn", report), "resolved", "classification"
        )

    def test_numeric_multiclass_codes_with_categorical_semantics(self):
        frame = pd.DataFrame({"disease_category": [0, 1, 2, 3] * 8})
        report = _report(
            "disease_category", semantic_type="categorical", semantic_role="category",
            value_function="classification", numeric_meaning="nominal_code",
        )
        self.assert_resolution(
            resolve_task_type(frame, "disease_category", report),
            "resolved", "classification",
        )

    def test_numeric_customer_id_is_unresolved(self):
        frame = pd.DataFrame({"customer_id": range(10001, 10031)})
        self.assert_resolution(
            resolve_task_type(frame, "customer_id"), "unresolved", None
        )

    def test_repeated_customer_id_is_still_unresolved(self):
        frame = pd.DataFrame({"customer_id": [10001, 10002, 10003] * 10})
        self.assert_resolution(
            resolve_task_type(frame, "customer_id"), "unresolved", None
        )

    def test_string_uuid_is_unresolved(self):
        frame = pd.DataFrame({"target": [str(uuid4()) for _ in range(12)]})
        result = resolve_task_type(frame, "target")
        self.assert_resolution(result, "unresolved", None)
        self.assertGreaterEqual(result.evidence["uuid_pattern_coverage"], 0.95)

    def test_constant_target_is_unresolved(self):
        self.assert_resolution(
            resolve_task_type(pd.DataFrame({"target": [7] * 10}), "target"),
            "unresolved", None,
        )

    def test_all_null_target_is_unresolved(self):
        self.assert_resolution(
            resolve_task_type(pd.DataFrame({"target": [None] * 10}), "target"),
            "unresolved", None,
        )

    def test_datetime_target_is_unresolved(self):
        frame = pd.DataFrame({"target": pd.date_range("2025-01-01", periods=10)})
        self.assert_resolution(resolve_task_type(frame, "target"), "unresolved", None)

    def test_string_date_target_is_unresolved(self):
        frame = pd.DataFrame({"target": [f"2025-01-{day:02d}" for day in range(1, 11)]})
        self.assert_resolution(resolve_task_type(frame, "target"), "unresolved", None)

    def test_free_form_text_is_unresolved(self):
        frame = pd.DataFrame({
            "target": [
                f"This is a different long narrative outcome number {index} with free form detail."
                for index in range(20)
            ]
        })
        self.assert_resolution(resolve_task_type(frame, "target"), "unresolved", None)

    def test_ambiguous_low_cardinality_integer_without_semantics_is_unresolved(self):
        frame = pd.DataFrame({"target": [1, 2, 3, 4] * 10})
        self.assert_resolution(resolve_task_type(frame, "target"), "unresolved", None)

    def test_conflicting_measurement_semantics_and_discrete_shape_is_unresolved(self):
        frame = pd.DataFrame({"score": [1, 2, 3] * 20})
        report = _report(
            "score", semantic_type="numeric_measurement", semantic_role="measurement",
            value_function="measurement", numeric_meaning="magnitude",
        )
        self.assert_resolution(resolve_task_type(frame, "score", report), "unresolved", None)

    def test_missing_values_with_clear_classification(self):
        frame = pd.DataFrame({"target": ["yes", "no", None, "yes", "no", None, "yes"]})
        result = resolve_task_type(frame, "target")
        self.assert_resolution(result, "resolved", "classification")
        self.assertEqual(result.evidence["non_null_count"], 5)

    def test_missing_values_with_clear_regression(self):
        frame = pd.DataFrame({"target": [0.1, 1.25, None, 2.7, 3.9, 5.15, 8.6]})
        result = resolve_task_type(frame, "target")
        self.assert_resolution(result, "resolved", "regression")
        self.assertEqual(result.evidence["missing_count"], 1)

    def test_missing_target_column_is_controlled_unresolved(self):
        result = resolve_task_type(pd.DataFrame({"feature": [1, 2, 3]}), "target")
        self.assert_resolution(result, "unresolved", None)
        self.assertEqual(result.evidence["target_column_occurrences"], 0)

    def test_duplicate_target_column_is_controlled_unresolved(self):
        frame = pd.DataFrame([[1, 2], [3, 4]], columns=["target", "target"])
        result = resolve_task_type(frame, "target")
        self.assert_resolution(result, "unresolved", None)
        self.assertEqual(result.evidence["target_column_occurrences"], 2)

    def test_pipeline1_model_object_and_guard_audit_are_extracted(self):
        class FakeModel:
            def model_dump(self, mode="python"):
                return {
                    "semantic_profile": {
                        "columns": [_report("target")["semantic_profile"]["columns"][0]],
                        "telemetry": {"semantic_contradiction_guard": {"entries": [{
                            "column_name": "target",
                            "contradiction_detected": True,
                            "contradiction_code": "example_guard",
                        }]}},
                    }
                }

        result = resolve_task_type(
            pd.DataFrame({"target": ["a", "b"] * 4}), "target", FakeModel()
        )
        self.assert_resolution(result, "resolved", "classification")
        self.assertTrue(result.evidence["semantic_contradiction_detected"])

    def test_nonnumeric_label_ignores_inapplicable_numeric_meaning(self):
        frame = pd.DataFrame({"target": ["yes", "no"] * 6})
        report = _report(
            "target", semantic_type="categorical", value_function="classification",
            numeric_meaning="identifier",
        )
        result = resolve_task_type(frame, "target", report)
        self.assert_resolution(result, "resolved", "classification")
        self.assertFalse(result.evidence["identifier_semantic_signal"])

    def test_conflicting_identifier_and_classification_semantics_is_unresolved(self):
        frame = pd.DataFrame({"target": ["yes", "no"] * 6})
        report = _report(
            "target", semantic_type="categorical", value_function="entity_reference",
            numeric_meaning="not_applicable",
        )
        result = resolve_task_type(frame, "target", report)
        self.assert_resolution(result, "unresolved", None)
        self.assertIn("conflicting", result.reason.casefold())


if __name__ == "__main__":
    unittest.main()
