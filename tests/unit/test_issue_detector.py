import unittest

import numpy as np
import pandas as pd

from pipelines.pipeline1_cleaning.profiling.dataset_profiler import profile_dataset
from pipelines.pipeline1_cleaning.quality.issue_detector import DetectionConfig, detect_issues


def detect(dataframe, **config_overrides):
    config = DetectionConfig(**config_overrides)
    return detect_issues(dataframe, profile_dataset(dataframe), config)


def by_type(issues, issue_type):
    return [issue for issue in issues if issue.issue_type == issue_type]


class IssueDetectorTests(unittest.TestCase):
    def test_missing_values_are_counted(self):
        issues = detect(pd.DataFrame({"value": [1.0, np.nan, 3.0]}))
        issue = by_type(issues, "missing_values")[0]
        self.assertEqual(issue.affected_count, 1)
        self.assertAlmostEqual(issue.affected_percentage, 33.3333)

    def test_repeated_column_values_are_not_duplicate_rows(self):
        issues = detect(pd.DataFrame({"group": ["A", "A", "A"], "sequence": [1, 2, 3]}))
        self.assertFalse(by_type(issues, "exact_duplicate_rows"))

    def test_exact_duplicate_rows_are_detected(self):
        issues = detect(pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]}))
        issue = by_type(issues, "exact_duplicate_rows")[0]
        self.assertEqual(issue.affected_count, 1)
        self.assertEqual(issue.target_columns, [])

    def test_whitespace_variants_are_detected_and_examples_are_capped(self):
        frame = pd.DataFrame({"value": [" x", "x ", "   ", " y ", "z", " q"]})
        issues = detect(frame, example_cap=2)
        issue = by_type(issues, "whitespace_issue")[0]
        self.assertEqual(issue.affected_count, 5)
        self.assertLessEqual(len(issue.examples), 2)

    def test_case_variant_candidate_uses_conservative_cardinality(self):
        values = ["Sales", "sales", " SALES "] * 10
        issue = by_type(detect(pd.DataFrame({"department": values})), "case_variant_candidate")[0]
        self.assertEqual(issue.status, "candidate")
        self.assertIn("sales", issue.evidence["variant_groups"])

    def test_mixed_numeric_representation_is_detected(self):
        values = ["100", "200", "bad", "350", "unknown", "400"]
        issue = by_type(
            detect(pd.DataFrame({"amount": values})),
            "mixed_numeric_representation_candidate",
        )[0]
        self.assertEqual(issue.evidence["parseable_count"], 4)

    def test_zero_padded_identifier_shape_is_guarded(self):
        values = ["001", "002", "003", "bad", "004", "005"]
        issues = detect(pd.DataFrame({"code": values}))
        self.assertFalse(by_type(issues, "mixed_numeric_representation_candidate"))

    def test_inconsistent_date_shapes_are_detected(self):
        values = ["2025-01-01", "2025/01/02", "2025-01-03", "2025/01/04", "other"]
        issue = by_type(
            detect(pd.DataFrame({"date_text": values})),
            "inconsistent_date_format_candidate",
        )[0]
        self.assertEqual(issue.status, "candidate")

    def test_infinities_are_not_counted_as_missing(self):
        issues = detect(pd.DataFrame({"value": [1.0, np.inf, -np.inf, np.nan]}))
        infinity = by_type(issues, "non_finite_numeric")[0]
        missing = by_type(issues, "missing_values")[0]
        self.assertEqual(infinity.affected_count, 2)
        self.assertEqual(missing.affected_count, 1)

    def test_iqr_outlier_is_a_candidate(self):
        values = list(range(1, 21)) + [1000]
        issue = by_type(
            detect(pd.DataFrame({"measurement": values})),
            "statistical_outlier_candidate",
        )[0]
        self.assertEqual(issue.status, "candidate")
        self.assertEqual(issue.affected_count, 1)

    def test_rare_category_is_an_observation(self):
        values = ["common"] * 98 + ["other"] + ["rare"]
        issue = by_type(
            detect(pd.DataFrame({"category": values})),
            "rare_category_observation",
        )[0]
        self.assertEqual(issue.status, "observation")
        self.assertEqual(issue.affected_count, 2)

    def test_simple_clean_input_has_no_obvious_issues(self):
        frame = pd.DataFrame({
            "category": ["A", "B"] * 10,
            "measurement": list(range(20)),
        })
        self.assertEqual(detect(frame), [])


if __name__ == "__main__":
    unittest.main()
