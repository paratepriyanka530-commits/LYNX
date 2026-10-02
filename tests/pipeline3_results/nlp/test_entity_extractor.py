from __future__ import annotations

import json
import unittest

from pipelines.pipeline3_results.nlp.entity_extractor import extract_entities
from pipelines.pipeline3_results.nlp.intent_classifier import classify_intent


class EntityExtractorTests(unittest.TestCase):
    def test_average_salary_by_department(self) -> None:
        result = extract_entities("show average salary by department")
        self.assertEqual(result.measure_terms, ["salary"])
        self.assertEqual(result.dimension_terms, ["department"])
        self.assertEqual(result.aggregation, "MEAN")
        self.assertIn("by_pattern", result.signals)
        self.assertIn("explicit_aggregation", result.signals)

    def test_question_scaffolding_is_removed_from_simple_terms(self) -> None:
        average = extract_entities("What is the average salary?")
        median = extract_entities("What is the median order value?")
        count = extract_entities("How many customers are there?")
        self.assertEqual(average.measure_terms, ["salary"])
        self.assertEqual(median.measure_terms, ["order value"])
        self.assertEqual(count.measure_terms, ["customers"])

    def test_numbered_comparative_question_preserves_limit(self) -> None:
        result = extract_entities(
            "Which 3 departments have the lowest average score?"
        )
        self.assertEqual(result.dimension_terms, ["departments"])
        self.assertEqual(result.measure_terms, ["score"])
        self.assertEqual(result.comparison, "BOTTOM_N")
        self.assertEqual(result.limit, 3)

    def test_total_sales_by_region(self) -> None:
        result = extract_entities("total sales by region")
        self.assertEqual(result.measure_terms, ["sales"])
        self.assertEqual(result.dimension_terms, ["region"])
        self.assertEqual(result.aggregation, "SUM")

    def test_revenue_over_time_preserves_time_without_inventing_chart(self) -> None:
        result = extract_entities("show revenue over time")
        self.assertEqual(result.measure_terms, ["revenue"])
        self.assertEqual(result.dimension_terms, [])
        self.assertEqual(result.time_terms, ["over time"])
        self.assertIsNone(result.requested_chart)
        self.assertIn("time_expression", result.signals)

    def test_relationship_between_two_measures(self) -> None:
        result = extract_entities("show relationship between discount and revenue")
        self.assertEqual(result.measure_terms, ["discount", "revenue"])
        self.assertEqual(result.dimension_terms, [])
        self.assertIn("between_pattern", result.signals)

    def test_comparative_question_keeps_comparison_separate(self) -> None:
        result = extract_entities("which department has the highest salary?")
        self.assertEqual(result.dimension_terms, ["department"])
        self.assertEqual(result.measure_terms, ["salary"])
        self.assertEqual(result.comparison, "HIGHEST")
        self.assertEqual(result.aggregation, "NONE")

    def test_top_numeric_and_word_limits(self) -> None:
        numeric = extract_entities("show top 5 products by revenue")
        word = extract_entities("show top five products by revenue")
        for result in (numeric, word):
            self.assertEqual(result.dimension_terms, ["products"])
            self.assertEqual(result.measure_terms, ["revenue"])
            self.assertEqual(result.comparison, "TOP_N")
            self.assertEqual(result.limit, 5)
            self.assertIn("top_n_pattern", result.signals)

    def test_bottom_n_with_independent_sum(self) -> None:
        result = extract_entities("bottom 10 customers by total spend")
        self.assertEqual(result.dimension_terms, ["customers"])
        self.assertEqual(result.measure_terms, ["spend"])
        self.assertEqual(result.aggregation, "SUM")
        self.assertEqual(result.comparison, "BOTTOM_N")
        self.assertEqual(result.limit, 10)

    def test_explicit_chart_only_when_named(self) -> None:
        bar = extract_entities("bar chart of sales by region")
        scatter = extract_entities("scatter plot of discount and revenue")
        implicit = extract_entities("show sales by region")
        self.assertEqual(bar.requested_chart, "BAR")
        self.assertEqual(bar.measure_terms, ["sales"])
        self.assertEqual(bar.dimension_terms, ["region"])
        self.assertEqual(scatter.requested_chart, "SCATTER")
        self.assertEqual(scatter.measure_terms, ["discount", "revenue"])
        self.assertIsNone(implicit.requested_chart)

    def test_named_filter(self) -> None:
        result = extract_entities("sales in north region")
        self.assertEqual(result.measure_terms, ["sales"])
        self.assertEqual(len(result.filters), 1)
        self.assertEqual(result.filters[0].column_term, "region")
        self.assertEqual(result.filters[0].value_term, "north")
        self.assertEqual(result.filters[0].operator, "EQUALS")
        self.assertIn("filter_pattern", result.signals)

    def test_where_filter(self) -> None:
        result = extract_entities("revenue where category is electronics")
        self.assertEqual(result.measure_terms, ["revenue"])
        self.assertEqual(result.filters[0].column_term, "category")
        self.assertEqual(result.filters[0].value_term, "electronics")

    def test_year_filter_and_time_expression(self) -> None:
        result = extract_entities("sales in 2025")
        self.assertEqual(result.measure_terms, ["sales"])
        self.assertEqual(result.time_terms, ["in 2025"])
        self.assertEqual(result.filters[0].value_term, "2025")
        self.assertIsNone(result.filters[0].column_term)

    def test_temporal_expressions(self) -> None:
        cases = {
            "show monthly revenue": ("revenue", "monthly"),
            "sales over month": ("sales", "over month"),
            "orders last year": ("orders", "last year"),
            "sales between 2024 and 2025": ("sales", "between 2024 and 2025"),
        }
        for text, (measure, time_term) in cases.items():
            with self.subTest(text=text):
                result = extract_entities(text)
                self.assertEqual(result.measure_terms, [measure])
                self.assertIn(time_term, result.time_terms)

    def test_other_grammatical_relations(self) -> None:
        cases = {
            "revenue across regions": (["revenue"], ["regions"]),
            "orders per category": (["orders"], ["category"]),
            "discount versus revenue": (["discount", "revenue"], []),
            "relationship between price and rating": (["price", "rating"], []),
        }
        for text, (measures, dimensions) in cases.items():
            with self.subTest(text=text):
                result = extract_entities(text)
                self.assertEqual(result.measure_terms, measures)
                self.assertEqual(result.dimension_terms, dimensions)

    def test_best_and_worst_comparisons(self) -> None:
        best = extract_entities("which product performed best")
        worst = extract_entities("which product performed worst")
        self.assertEqual(best.dimension_terms, ["product"])
        self.assertEqual(best.comparison, "HIGHEST")
        self.assertEqual(worst.dimension_terms, ["product"])
        self.assertEqual(worst.comparison, "LOWEST")

    def test_smallest_comparative_is_not_min_aggregation(self) -> None:
        result = extract_entities("which department has the smallest salary")
        self.assertEqual(result.comparison, "LOWEST")
        self.assertEqual(result.aggregation, "NONE")
        self.assertEqual(result.dimension_terms, ["department"])
        self.assertEqual(result.measure_terms, ["salary"])

    def test_vague_requests_do_not_fabricate_entities(self) -> None:
        for text in ("show me something interesting", "plot this"):
            with self.subTest(text=text):
                result = extract_entities(text)
                self.assertEqual(result.measure_terms, [])
                self.assertEqual(result.dimension_terms, [])

    def test_optional_intent_is_audited_but_does_not_create_entities(self) -> None:
        classified = classify_intent("plot this")
        result = extract_entities("plot this", intent=classified)
        self.assertEqual(result.measure_terms, [])
        self.assertIn("intent_context:visualization", result.signals)

    def test_no_column_resolution_and_json_safety(self) -> None:
        result = extract_entities("sales by region")
        self.assertEqual(result.measure_terms, ["sales"])
        self.assertEqual(result.dimension_terms, ["region"])
        payload = result.model_dump(mode="json")
        self.assertEqual(json.loads(json.dumps(payload)), payload)
        serialized = json.dumps(payload)
        self.assertNotIn("total_price", serialized)
        self.assertNotIn("customer_region", serialized)

    def test_repeated_extraction_is_deterministic(self) -> None:
        text = "bottom 10 customers by total spend"
        expected = extract_entities(text)
        for _ in range(25):
            self.assertEqual(extract_entities(text), expected)

    def test_input_validation(self) -> None:
        with self.assertRaises(TypeError):
            extract_entities(42)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            extract_entities("sales", intent="BAD")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
