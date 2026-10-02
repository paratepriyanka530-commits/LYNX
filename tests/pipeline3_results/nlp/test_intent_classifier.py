from __future__ import annotations

import json
import unittest

from pydantic import ValidationError

from pipelines.pipeline3_results.nlp.intent_classifier import classify_intent


class IntentClassifierTests(unittest.TestCase):
    def assert_intent(self, text: str, expected: str, context: str | None = None) -> None:
        result = classify_intent(text, context=context)
        self.assertEqual(result.intent, expected)
        self.assertGreater(len(result.signals), 0)

    def test_visualization_examples(self) -> None:
        examples = (
            "make a bar chart of salary by department",
            "plot revenue over time",
            "show sales by region",
            "show relationship between discount and revenue",
            "graph discount against revenue",
            "show me the distribution of age",
            "visualize product category counts",
            "compare revenue across regions",
            "show monthly sales",
        )
        for text in examples:
            with self.subTest(text=text):
                self.assert_intent(text, "VISUALIZATION")

    def test_insight_examples(self) -> None:
        examples = (
            "which department has the highest salary?",
            "what is the average revenue?",
            "how many customers are there?",
            "which region generated the most sales?",
            "what category has the lowest revenue?",
            "what is the median order value?",
            "which product performed best?",
            "what was total sales?",
        )
        for text in examples:
            with self.subTest(text=text):
                self.assert_intent(text, "INSIGHT")

    def test_unknown_examples(self) -> None:
        examples = (
            "tell me something",
            "analyze this",
            "show interesting stuff",
            "what can you do?",
            "help me understand the data",
        )
        for text in examples:
            with self.subTest(text=text):
                result = classify_intent(text)
                self.assertEqual(result.intent, "UNKNOWN")
                self.assertEqual(result.confidence, "LOW")

    def test_visualization_context_disambiguates_by_fragment(self) -> None:
        result = classify_intent("sales by region", context="visualization")
        self.assertEqual(result.intent, "VISUALIZATION")
        self.assertEqual(result.confidence, "LOW")
        self.assertTrue(result.context_used)
        self.assertIn("context_fallback", result.signals)

    def test_insights_context_does_not_invent_question_structure(self) -> None:
        result = classify_intent("sales by region", context="insights")
        self.assertEqual(result.intent, "UNKNOWN")
        self.assertFalse(result.context_used)
        self.assertIn("insufficient_intent_evidence", result.signals)

    def test_insights_context_can_route_an_analytical_fragment(self) -> None:
        result = classify_intent("average revenue", context="insights")
        self.assertEqual(result.intent, "INSIGHT")
        self.assertEqual(result.confidence, "LOW")
        self.assertTrue(result.context_used)

    def test_explicit_visualization_overrides_insights_context(self) -> None:
        result = classify_intent("plot sales by region", context="insights")
        self.assertEqual(result.intent, "VISUALIZATION")
        self.assertEqual(result.confidence, "HIGH")
        self.assertFalse(result.context_used)
        self.assertIn("explicit_chart_term", result.signals)

    def test_explicit_insight_overrides_visualization_context(self) -> None:
        result = classify_intent(
            "which region has the highest sales?", context="visualization"
        )
        self.assertEqual(result.intent, "INSIGHT")
        self.assertEqual(result.confidence, "HIGH")
        self.assertFalse(result.context_used)
        self.assertIn("comparative_question", result.signals)

    def test_show_highest_by_remains_visualization(self) -> None:
        result = classify_intent("show highest sales by region")
        self.assertEqual(result.intent, "VISUALIZATION")
        self.assertIn("show_by_pattern", result.signals)

    def test_result_is_frozen_and_json_safe(self) -> None:
        result = classify_intent("plot revenue over time")
        payload = result.model_dump(mode="json")
        self.assertEqual(json.loads(json.dumps(payload)), payload)
        with self.assertRaises(ValidationError):
            result.intent = "UNKNOWN"  # type: ignore[misc]

    def test_repeated_classification_is_deterministic(self) -> None:
        expected = classify_intent("which region generated the most sales?")
        for _ in range(25):
            self.assertEqual(
                classify_intent("which region generated the most sales?"), expected
            )

    def test_input_and_context_validation(self) -> None:
        with self.assertRaises(TypeError):
            classify_intent(42)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            classify_intent("sales", context=42)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            classify_intent("sales", context="other")

    def test_context_is_case_and_whitespace_normalized(self) -> None:
        result = classify_intent("sales by region", context=" Visualization ")
        self.assertEqual(result.intent, "VISUALIZATION")
        self.assertTrue(result.context_used)


if __name__ == "__main__":
    unittest.main()
