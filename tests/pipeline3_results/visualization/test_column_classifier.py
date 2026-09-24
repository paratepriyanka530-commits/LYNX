"""Unit tests for Pipeline 3 Deterministic Column Intelligence."""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from pipelines.pipeline3_results.visualization.column_classifier import (
    classify_column,
    classify_columns,
    classify_dataframe,
)
from pipelines.pipeline3_results.visualization.schemas import (
    ALL_ROLES,
    ColumnClassification,
    DatasetClassification,
)


class TestMeasureClassification(unittest.TestCase):
    """Test classification of MEASURE columns."""

    def test_continuous_floats(self) -> None:
        s = pd.Series([10.5, 23.4, 55.1, 88.9, 102.3], name="measurement")
        res = classify_column(s)
        self.assertEqual(res.role, "MEASURE")
        self.assertIn("numeric_float_continuous_values", res.signals)

    def test_measure_name_signals(self) -> None:
        for name in ["sales", "revenue", "price", "quantity", "cost", "salary", "age", "score"]:
            s = pd.Series([10, 20, 30, 40, 50], name=name)
            res = classify_column(s)
            self.assertEqual(res.role, "MEASURE", f"Failed for name {name}")
            self.assertIn("measure_name_signal", res.signals)

    def test_high_cardinality_integers(self) -> None:
        s = pd.Series(list(range(50, 100)), name="metric_data")
        res = classify_column(s)
        self.assertEqual(res.role, "MEASURE")
        self.assertIn("numeric_high_cardinality_measure", res.signals)

    def test_duration_is_measure_not_temporal(self) -> None:
        s = pd.Series([12.5, 45.0, 90.2, 120.4], name="time_spent")
        res = classify_column(s)
        self.assertEqual(res.role, "MEASURE")
        self.assertNotEqual(res.role, "TEMPORAL")

    def test_duration_seconds_is_measure(self) -> None:
        s = pd.Series([30, 60, 120, 240, 480], name="duration_seconds")
        res = classify_column(s)
        self.assertEqual(res.role, "MEASURE")

    def test_measure_with_pipeline1_evidence(self) -> None:
        s = pd.Series([1, 2, 3, 4], name="val")
        p1_evidence = {
            "semantic_type": "continuous_quantity",
            "value_function": "continuous_quantity",
            "numeric_meaning": "continuous_quantity",
        }
        res = classify_column(s, semantic_evidence=p1_evidence)
        self.assertEqual(res.role, "MEASURE")
        self.assertIn("pipeline1_measure_semantic", res.signals)


class TestCategoryClassification(unittest.TestCase):
    """Test classification of CATEGORY columns."""

    def test_string_categories_with_repetition(self) -> None:
        s = pd.Series(["East", "West", "East", "North", "South", "West", "East"], name="region")
        res = classify_column(s)
        self.assertEqual(res.role, "CATEGORY")
        self.assertIn("bounded_cardinality_category", res.signals)
        self.assertIn("category_name_signal", res.signals)

    def test_discrete_numeric_low_cardinality(self) -> None:
        s = pd.Series([1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 1, 3], name="code")
        res = classify_column(s)
        self.assertEqual(res.role, "CATEGORY")
        self.assertIn("discrete_numeric_low_cardinality_category", res.signals)

    def test_discrete_numeric_with_category_name(self) -> None:
        s = pd.Series([1, 2, 3, 4], name="tier")
        res = classify_column(s)
        self.assertEqual(res.role, "CATEGORY")
        self.assertIn("category_name_signal", res.signals)

    def test_category_with_pipeline1_evidence(self) -> None:
        s = pd.Series(["A", "B", "C"], name="alpha_group")
        p1_evidence = {"semantic_type": "category", "value_function": "category_nominal"}
        res = classify_column(s, semantic_evidence=p1_evidence)
        self.assertEqual(res.role, "CATEGORY")
        self.assertIn("pipeline1_category_semantic", res.signals)


class TestConservativeNumericZeroOneRule(unittest.TestCase):
    """Test user requirement: numeric {0, 1} columns must NOT be BOOLEAN unless

    corroborated by boolean naming or Pipeline 1 evidence.
    """

    def test_numeric_zero_one_without_boolean_context_is_category(self) -> None:
        for name in ["tier", "group", "cluster", "code", "x", "label_class"]:
            s = pd.Series([0, 1, 0, 1, 1, 0, 0, 1], name=name)
            res = classify_column(s)
            self.assertEqual(
                res.role,
                "CATEGORY",
                f"Column {name} with {0, 1} should be CATEGORY, got {res.role}",
            )
            self.assertIn("no_boolean_semantics_fallback_category", res.signals)

    def test_numeric_zero_one_with_boolean_name_is_boolean(self) -> None:
        for name in ["is_active", "has_child", "churned", "defaulted", "flag", "opt_in", "approved"]:
            s = pd.Series([0, 1, 0, 1, 1, 0, 0, 1], name=name)
            res = classify_column(s)
            self.assertEqual(
                res.role,
                "BOOLEAN",
                f"Column {name} with boolean name should be BOOLEAN, got {res.role}",
            )
            self.assertIn("boolean_name_signal", res.signals)

    def test_numeric_zero_one_with_pipeline1_evidence_is_boolean(self) -> None:
        s = pd.Series([0, 1, 0, 1], name="generic_status")
        p1_evidence = {"value_function": "boolean_state", "semantic_type": "boolean"}
        res = classify_column(s, semantic_evidence=p1_evidence)
        self.assertEqual(res.role, "BOOLEAN")
        self.assertIn("pipeline1_boolean_semantic", res.signals)


class TestTemporalClassification(unittest.TestCase):
    """Test classification of TEMPORAL columns."""

    def test_physical_datetime_dtype(self) -> None:
        s = pd.Series(pd.date_range("2023-01-01", periods=5, freq="D"), name="event_time")
        res = classify_column(s)
        self.assertEqual(res.role, "TEMPORAL")
        self.assertIn("datetime_dtype", res.signals)

    def test_string_iso_date_with_temporal_name(self) -> None:
        s = pd.Series(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"], name="order_date")
        res = classify_column(s)
        self.assertEqual(res.role, "TEMPORAL")
        self.assertIn("temporal_name_signal", res.signals)
        self.assertIn("date_parseable_format", res.signals)

    def test_integer_year_with_year_name(self) -> None:
        s = pd.Series([2019, 2020, 2021, 2022, 2023], name="sales_year")
        res = classify_column(s)
        self.assertEqual(res.role, "TEMPORAL")
        self.assertIn("year_integer_range", res.signals)

    def test_epoch_timestamp(self) -> None:
        s = pd.Series([1672531200, 1672531260, 1672531320], name="event_timestamp")
        res = classify_column(s)
        self.assertEqual(res.role, "TEMPORAL")
        self.assertIn("epoch_timestamp", res.signals)


class TestConservativeTemporalRule(unittest.TestCase):
    """Test user requirement: strong date parsing alone must NOT make an arbitrary

    string column TEMPORAL unless corroborated by temporal naming or Pipeline 1 evidence.
    """

    def test_date_string_without_temporal_name_is_not_temporal(self) -> None:
        # Values look like dates, but column name has no temporal hints and no P1 evidence
        s = pd.Series(["2023-01-01", "2023-01-02", "2023-01-03"], name="tracking_code")
        res = classify_column(s)
        self.assertNotEqual(
            res.role,
            "TEMPORAL",
            "Arbitrary string column with date format should not be TEMPORAL without naming/semantic corroboration",
        )

    def test_date_string_with_pipeline1_temporal_evidence_is_temporal(self) -> None:
        s = pd.Series(["2023-01-01", "2023-01-02", "2023-01-03"], name="tracking_code")
        p1_evidence = {"semantic_type": "date", "value_function": "temporal"}
        res = classify_column(s, semantic_evidence=p1_evidence)
        self.assertEqual(res.role, "TEMPORAL")
        self.assertIn("pipeline1_temporal_semantic", res.signals)


class TestIdentifierClassification(unittest.TestCase):
    """Test classification of IDENTIFIER columns."""

    def test_uuid_pattern_strings(self) -> None:
        s = pd.Series(
            [
                "c9bf9e57-1685-4c89-bafb-ff5af830be8a",
                "e3b0c442-98fc-1c14-9afb-4c0000000000",
                "8d8b9e57-1685-4c89-bafb-ff5af830be8b",
            ],
            name="record_guid",
        )
        res = classify_column(s)
        self.assertEqual(res.role, "IDENTIFIER")
        self.assertIn("uuid_pattern_match", res.signals)

    def test_identifier_name_with_high_cardinality(self) -> None:
        for name in ["customer_id", "order_id", "user_id", "transaction_key", "account_id"]:
            s = pd.Series([f"ID_{i}" for i in range(20)], name=name)
            res = classify_column(s)
            self.assertEqual(res.role, "IDENTIFIER", f"Failed for {name}")
            self.assertIn("identifier_name_signal", res.signals)

    def test_identifier_with_pipeline1_evidence(self) -> None:
        s = pd.Series(["X1", "X2", "X3", "X4"], name="custom_col")
        p1_evidence = {"semantic_type": "identifier", "value_function": "identifier"}
        res = classify_column(s, semantic_evidence=p1_evidence)
        self.assertEqual(res.role, "IDENTIFIER")
        self.assertIn("pipeline1_identifier_semantic", res.signals)


class TestDescriptiveTextClassification(unittest.TestCase):
    """Test classification of DESCRIPTIVE_TEXT columns."""

    def test_long_free_text_with_name_signal(self) -> None:
        s = pd.Series(
            [
                "The product worked well and was easy to assemble.",
                "Customer requested a full refund due to shipping damage.",
                "Fast delivery and great customer support throughout.",
            ],
            name="customer_review",
        )
        res = classify_column(s)
        self.assertEqual(res.role, "DESCRIPTIVE_TEXT")
        self.assertIn("descriptive_name_signal", res.signals)

    def test_free_text_without_name_signal(self) -> None:
        s = pd.Series(
            [
                "Quick brown fox jumps over the lazy dog repeatedly.",
                "Detailed summary of events occurring during second quarter.",
                "Thorough investigation revealed multiple discrepancies in records.",
                "Additional commentary provided by independent third party auditors.",
            ],
            name="notes_field",
        )
        res = classify_column(s)
        self.assertEqual(res.role, "DESCRIPTIVE_TEXT")

    def test_descriptive_text_with_pipeline1_evidence(self) -> None:
        s = pd.Series(["Summary text here.", "Another summary text."], name="col")
        p1_evidence = {"semantic_type": "narrative", "value_function": "narrative_text"}
        res = classify_column(s, semantic_evidence=p1_evidence)
        self.assertEqual(res.role, "DESCRIPTIVE_TEXT")
        self.assertIn("pipeline1_text_semantic", res.signals)


class TestBooleanClassification(unittest.TestCase):
    """Test classification of BOOLEAN columns."""

    def test_physical_bool_dtype(self) -> None:
        s = pd.Series([True, False, True, True], name="flag_status")
        res = classify_column(s)
        self.assertEqual(res.role, "BOOLEAN")
        self.assertIn("boolean_dtype", res.signals)

    def test_string_boolean_values(self) -> None:
        s = pd.Series(["true", "false", "true", "false"], name="verified")
        res = classify_column(s)
        self.assertEqual(res.role, "BOOLEAN")
        self.assertIn("string_boolean_values", res.signals)

    def test_yes_no_strings(self) -> None:
        s = pd.Series(["yes", "no", "yes", "no"], name="opt_in")
        res = classify_column(s)
        self.assertEqual(res.role, "BOOLEAN")
        self.assertIn("string_boolean_values", res.signals)


class TestEdgeCasesAndAmbiguity(unittest.TestCase):
    """Test edge cases, missing values, constants, and empty data."""

    def test_all_missing_values(self) -> None:
        s = pd.Series([None, np.nan, None], name="all_null_col")
        res = classify_column(s)
        self.assertEqual(res.role, "UNKNOWN")
        self.assertIn("all_values_missing", res.signals)
        self.assertEqual(res.missing_count, 3)
        self.assertEqual(res.missing_ratio, 1.0)

    def test_empty_series(self) -> None:
        s = pd.Series([], dtype=object, name="empty_col")
        res = classify_column(s)
        self.assertEqual(res.role, "UNKNOWN")
        self.assertIn("empty_column", res.signals)

    def test_constant_value_zero_variance(self) -> None:
        s = pd.Series([42, 42, 42, 42], name="constant_number")
        res = classify_column(s)
        self.assertEqual(res.role, "UNKNOWN")
        self.assertIn("constant_single_value", res.signals)
        self.assertIn("insufficient_variance", res.signals)

    def test_empty_dataframe(self) -> None:
        df = pd.DataFrame()
        res = classify_columns(df)
        self.assertEqual(res.row_count, 0)
        self.assertEqual(res.column_count, 0)
        self.assertEqual(len(res.columns), 0)
        for role in ALL_ROLES:
            self.assertEqual(res.role_counts[role], 0)


class TestDatasetClassification(unittest.TestCase):
    """Test full dataset classification and summary schema."""

    def setUp(self) -> None:
        self.df = pd.DataFrame(
            {
                "customer_id": [f"CUST_{i:03d}" for i in range(10)],
                "sales_amount": [10.5 * (i + 1) for i in range(10)],
                "region": ["North", "South", "East", "West", "North", "South", "East", "West", "North", "South"],
                "signup_date": [f"2023-01-{i+1:02d}" for i in range(10)],
                "is_subscribed": [1, 0, 1, 0, 1, 1, 0, 0, 1, 0],
                "feedback": [f"Feedback review comment number {i} with useful details." for i in range(10)],
                "constant_val": [999] * 10,
            }
        )

    def test_dataset_classification_all_roles_present(self) -> None:
        res = classify_columns(self.df, dataset_name="customer_orders")
        self.assertEqual(res.dataset_name, "customer_orders")
        self.assertEqual(res.row_count, 10)
        self.assertEqual(res.column_count, 7)

        # Check role assignments
        self.assertEqual(res.get_column("customer_id").role, "IDENTIFIER")
        self.assertEqual(res.get_column("sales_amount").role, "MEASURE")
        self.assertEqual(res.get_column("region").role, "CATEGORY")
        self.assertEqual(res.get_column("signup_date").role, "TEMPORAL")
        self.assertEqual(res.get_column("is_subscribed").role, "BOOLEAN")
        self.assertEqual(res.get_column("feedback").role, "DESCRIPTIVE_TEXT")
        self.assertEqual(res.get_column("constant_val").role, "UNKNOWN")

        # Check role counts match
        self.assertEqual(res.role_counts["IDENTIFIER"], 1)
        self.assertEqual(res.role_counts["MEASURE"], 1)
        self.assertEqual(res.role_counts["CATEGORY"], 1)
        self.assertEqual(res.role_counts["TEMPORAL"], 1)
        self.assertEqual(res.role_counts["BOOLEAN"], 1)
        self.assertEqual(res.role_counts["DESCRIPTIVE_TEXT"], 1)
        self.assertEqual(res.role_counts["UNKNOWN"], 1)

        # Check helper properties
        self.assertEqual(res.identifiers, ["customer_id"])
        self.assertEqual(res.measures, ["sales_amount"])
        self.assertEqual(res.categories, ["region"])
        self.assertEqual(res.temporals, ["signup_date"])
        self.assertEqual(res.booleans, ["is_subscribed"])
        self.assertEqual(res.descriptive_texts, ["feedback"])
        self.assertEqual(res.unknowns, ["constant_val"])

    def test_classify_dataframe_alias(self) -> None:
        res1 = classify_columns(self.df)
        res2 = classify_dataframe(self.df)
        self.assertEqual(res1.role_counts, res2.role_counts)
        self.assertEqual([c.role for c in res1.columns], [c.role for c in res2.columns])

    def test_determinism_and_reproducibility(self) -> None:
        res1 = classify_columns(self.df)
        res2 = classify_columns(self.df)
        self.assertEqual(res1.model_dump(), res2.model_dump())


class TestPipeline1ReportIntegration(unittest.TestCase):
    """Test extraction and handling of Pipeline 1 cleaning reports."""

    def test_full_cleaning_report_handling(self) -> None:
        df = pd.DataFrame(
            {
                "feat_x": [10.2, 20.4, 30.1],
                "feat_y": [1, 2, 3],
            }
        )
        report = {
            "semantic_profile": {
                "columns": [
                    {
                        "column_name": "feat_x",
                        "semantic_type": "continuous_quantity",
                        "value_function": "continuous_quantity",
                    },
                    {
                        "column_name": "FEAT_Y",  # case-insensitive check
                        "semantic_type": "identifier",
                        "value_function": "identifier",
                    },
                ]
            }
        }
        res = classify_columns(df, pipeline1_report=report)
        self.assertEqual(res.get_column("feat_x").role, "MEASURE")
        self.assertEqual(res.get_column("feat_y").role, "IDENTIFIER")

    def test_partial_and_missing_report_toleration(self) -> None:
        df = pd.DataFrame({"col_a": [1, 2, 3], "col_b": ["x", "y", "z"]})
        # Empty report
        res_empty = classify_columns(df, pipeline1_report={})
        self.assertIsNotNone(res_empty)
        # None report
        res_none = classify_columns(df, pipeline1_report=None)
        self.assertIsNotNone(res_none)
        # Malformed report
        res_malformed = classify_columns(df, pipeline1_report={"invalid_key": 123})
        self.assertIsNotNone(res_malformed)


if __name__ == "__main__":
    unittest.main()

