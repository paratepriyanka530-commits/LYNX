import json
import unittest

import pandas as pd

from api.dataset_overview import (
    MAX_PREVIEW_COLUMNS,
    MAX_PREVIEW_ROWS,
    build_dataset_overview,
    build_raw_data_preview,
    measure_memory_footprint_bytes,
)
from api.schemas import DatasetOverviewMetadata


class DatasetOverviewMetadataTests(unittest.TestCase):
    def test_memory_footprint_matches_deep_pandas_measurement_with_index(self):
        frame = pd.DataFrame({
            "amount": [1.5, 2.5, 3.5],
            "label": ["short", "a much longer object value", "short"],
        })
        expected = int(frame.memory_usage(index=True, deep=True).sum())
        measured = measure_memory_footprint_bytes(frame)
        self.assertIsInstance(measured, int)
        self.assertEqual(measured, expected)

    def test_empty_dataframe_memory_is_numeric_and_includes_index(self):
        frame = pd.DataFrame({
            "number": pd.Series(dtype="int64"),
            "label": pd.Series(dtype="object"),
        })
        self.assertEqual(
            measure_memory_footprint_bytes(frame),
            int(frame.memory_usage(index=True, deep=True).sum()),
        )

    def test_preview_preserves_order_and_first_ten_rows(self):
        frame = pd.DataFrame({
            "third": range(12),
            "first": [f"row-{index}" for index in range(12)],
            "second": range(100, 112),
        })
        preview = build_raw_data_preview(frame, privacy_clearance_passed=True)
        self.assertEqual(preview["columns"], ["third", "first", "second"])
        self.assertEqual(preview["returned_row_count"], MAX_PREVIEW_ROWS)
        self.assertEqual(preview["rows"][0], [0, "row-0", 100])
        self.assertEqual(preview["rows"][-1], [9, "row-9", 109])

    def test_preview_returns_all_rows_when_dataset_has_fewer_than_ten(self):
        frame = pd.DataFrame({"value": [4, 5, 6]})
        preview = build_raw_data_preview(frame, privacy_clearance_passed=True)
        self.assertEqual(preview["rows"], [[4], [5], [6]])
        self.assertEqual(preview["returned_row_count"], 3)

    def test_missing_and_datetime_values_are_strict_json(self):
        frame = pd.DataFrame({
            "when": [pd.Timestamp("2026-10-11T08:30:00"), pd.NaT],
            "amount": [1.0, float("nan")],
        })
        preview = build_raw_data_preview(frame, privacy_clearance_passed=True)
        self.assertEqual(preview["rows"][0], ["2026-10-11T08:30:00", 1.0])
        self.assertEqual(preview["rows"][1], [None, None])
        json.dumps(preview, allow_nan=False)

    def test_preview_does_not_modify_original_dataframe(self):
        frame = pd.DataFrame({"name": [" Alice ", None], "score": [1, 2]})
        original = frame.copy(deep=True)
        build_dataset_overview(
            frame,
            privacy_clearance_passed=True,
            pii_columns=["name"],
        )
        pd.testing.assert_frame_equal(frame, original)

    def test_privacy_columns_are_redacted_and_failed_clearance_withholds_all_rows(self):
        frame = pd.DataFrame({
            "contact_email": ["alice@example.com", None],
            "amount": [10, 20],
        })
        safe = build_raw_data_preview(
            frame,
            privacy_clearance_passed=True,
            pii_columns=["contact_email"],
        )
        self.assertEqual(safe["privacy_redacted_columns"], ["contact_email"])
        self.assertEqual(safe["rows"], [["[REDACTED]", 10], [None, 20]])

        blocked = build_raw_data_preview(
            frame,
            privacy_clearance_passed=False,
            pii_columns=["contact_email"],
        )
        self.assertFalse(blocked["available"])
        self.assertEqual(blocked["unavailable_reason"], "privacy_clearance_not_passed")
        self.assertEqual(blocked["columns"], [])
        self.assertEqual(blocked["rows"], [])

    def test_wide_preview_is_bounded_and_schema_serializes_byte_count(self):
        frame = pd.DataFrame({f"column_{index}": [index] for index in range(105)})
        overview = build_dataset_overview(frame, privacy_clearance_passed=True)
        preview = overview["raw_data_preview"]
        self.assertEqual(len(preview["columns"]), MAX_PREVIEW_COLUMNS)
        self.assertTrue(preview["columns_truncated"])
        validated = DatasetOverviewMetadata.model_validate(overview)
        dumped = validated.model_dump(mode="json")
        self.assertIsInstance(dumped["memory_footprint_bytes"], int)
        json.dumps(dumped, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
