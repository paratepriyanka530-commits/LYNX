from __future__ import annotations

import unittest

from pipelines.pipeline3_results.nlp.normalizer import (
    normalize_aggregation,
    normalize_chart,
    normalize_comparison,
    normalize_language,
    normalize_text,
    tokenize,
)


class NormalizerTests(unittest.TestCase):
    def test_aggregation_synonyms(self) -> None:
        expected = {
            "average": "MEAN", "avg": "MEAN", "mean": "MEAN",
            "total": "SUM", "sum": "SUM", "combined": "SUM",
            "count": "COUNT", "number of": "COUNT", "how many": "COUNT",
            "minimum": "MIN", "min": "MIN", "smallest": "MIN",
            "maximum": "MAX", "max": "MAX", "largest": "MAX",
            "median": "MEDIAN",
        }
        for phrase, aggregation in expected.items():
            with self.subTest(phrase=phrase):
                self.assertEqual(normalize_aggregation(phrase), aggregation)

    def test_comparison_concepts_are_exposed_separately(self) -> None:
        self.assertEqual(normalize_comparison("highest salary"), "HIGHEST")
        self.assertEqual(normalize_comparison("top products"), "HIGHEST")
        self.assertEqual(normalize_comparison("lowest score"), "LOWEST")
        self.assertEqual(normalize_comparison("bottom categories"), "LOWEST")
        # Later reasoning, not normalization, decides whether this overlap is
        # aggregation or comparison semantics.
        concepts = normalize_language("department with smallest salary")
        self.assertEqual(concepts["aggregation"], "MIN")
        self.assertEqual(concepts["comparison"], "LOWEST")

    def test_maximum_is_not_silently_reinterpreted_as_highest(self) -> None:
        concepts = normalize_language("maximum salary")
        self.assertEqual(concepts["aggregation"], "MAX")
        self.assertEqual(concepts["comparison"], "NONE")

    def test_chart_synonyms(self) -> None:
        expected = {
            "bar graph": "BAR", "bar chart": "BAR", "column chart": "BAR",
            "line graph": "LINE", "line chart": "LINE", "trend": "LINE",
            "scatter plot": "SCATTER", "scatter chart": "SCATTER",
            "histogram": "HISTOGRAM", "frequency chart": "COUNT_BAR",
            "count chart": "COUNT_BAR",
        }
        for phrase, chart in expected.items():
            with self.subTest(phrase=phrase):
                self.assertEqual(normalize_chart(phrase), chart)

    def test_punctuation_separator_and_whitespace_normalization(self) -> None:
        self.assertEqual(
            normalize_text("  Average, EMPLOYEE_salary---by\tdepartment_name?!  "),
            "average employee salary by department name",
        )
        self.assertEqual(
            tokenize("Revenue_by-region!"),
            ["revenue", "by", "region"],
        )

    def test_first_concept_is_selected_deterministically(self) -> None:
        text = "sum revenue then average cost"
        expected = normalize_language(text)
        self.assertEqual(expected["aggregation"], "SUM")
        for _ in range(20):
            self.assertEqual(normalize_language(text), expected)

    def test_normalization_does_not_resolve_or_classify_columns(self) -> None:
        result = normalize_language(
            "average employee_salary by department_name"
        )
        self.assertEqual(result["aggregation"], "MEAN")
        self.assertEqual(
            result["normalized_text"],
            "average employee salary by department name",
        )
        self.assertNotIn("measure", result)
        self.assertNotIn("dimension", result)
        self.assertNotIn("resolved_column", result)

    def test_absent_concepts_use_safe_defaults(self) -> None:
        result = normalize_language("salary by department")
        self.assertEqual(result["aggregation"], "NONE")
        self.assertEqual(result["comparison"], "NONE")
        self.assertIsNone(result["requested_chart"])

    def test_non_string_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            normalize_text(123)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
