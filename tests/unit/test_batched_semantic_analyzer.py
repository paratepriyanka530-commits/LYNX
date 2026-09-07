import unittest

from pydantic import ValidationError

from pipelines.pipeline1_cleaning.semantics.batched_analyzer import (
    BatchCoverageError,
    BatchSemanticConfig,
    analyze_batched_semantics,
    create_batches,
    validate_batch_coverage,
)
from pipelines.pipeline1_cleaning.semantics.batched_schema import BatchedColumnSemantic


def semantic(name):
    return BatchedColumnSemantic(
        column_name=name, semantic_type="unknown", value_function="classification",
        numeric_meaning="not_applicable",
        concept_description=f"Values represented by {name}.", semantic_role="unknown",
        confidence=0.8, reasoning_summary="Mock evidence.",
    )


class MockRetriever:
    def retrieve(self, *args, **kwargs):
        return []


class MockProvider:
    last_response_metadata = {}

    def __init__(self, batch_outputs):
        self.batch_outputs = list(batch_outputs)
        self.batch_prompts = []

    def generate_structured(self, prompt, response_model):
        if response_model.__name__ == "DatasetSemanticInference":
            return {
                "dataset_domain": None, "dataset_description": "Transaction records.",
                "row_entity": "transaction", "confidence": 0.9,
                "reasoning_summary": "Rows describe transactions.",
            }
        self.batch_prompts.append(prompt)
        output = self.batch_outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return {"columns": [item.model_dump() for item in output]}


def profile(names):
    return {
        "column_profiles": [
            {
                "column_name": name, "physical_dtype": "object",
                "missing": {"percentage": 0.0},
                "uniqueness": {"unique_ratio": 0.5},
                "repetition": {"repetition_rate": 0.5},
                "sample_values": ["A", "B"],
            }
            for name in names
        ]
    }


class BatchedSemanticAnalyzerTests(unittest.TestCase):
    def test_numeric_meaning_vocabulary_is_closed_and_required(self):
        for value in (
            "magnitude", "ordinal", "nominal_code", "identifier",
            "not_applicable", "unknown",
        ):
            item = semantic("value").model_copy(update={"numeric_meaning": value})
            self.assertEqual(item.numeric_meaning, value)
        payload = semantic("value").model_dump()
        payload.pop("numeric_meaning")
        with self.assertRaises(ValidationError):
            BatchedColumnSemantic.model_validate(payload)
        with self.assertRaises(ValidationError):
            BatchedColumnSemantic.model_validate({**payload, "numeric_meaning": "continuous"})

    def test_batch_creation(self):
        batches = create_batches([{"column_name": str(i)} for i in range(25)], 12)
        self.assertEqual([len(batch) for batch in batches], [12, 12, 1])

    def test_coverage_matches_by_name_not_position(self):
        matched = validate_batch_coverage(["a", "b"], [semantic("b"), semantic("a")])
        self.assertEqual(list(matched), ["a", "b"])

    def test_missing_output_is_rejected(self):
        with self.assertRaisesRegex(BatchCoverageError, "missing=b"):
            validate_batch_coverage(["a", "b"], [semantic("a")])

    def test_duplicate_output_is_rejected(self):
        with self.assertRaisesRegex(BatchCoverageError, "duplicate=a"):
            validate_batch_coverage(["a"], [semantic("a"), semantic("a")])

    def test_invented_output_is_rejected(self):
        with self.assertRaisesRegex(BatchCoverageError, "invented=x"):
            validate_batch_coverage(["a"], [semantic("a"), semantic("x")])

    def test_context_is_shared_and_results_are_ordered(self):
        provider = MockProvider([
            [semantic("b"), semantic("a")],
            [semantic("c")],
        ])
        result = analyze_batched_semantics(
            profile(["a", "b", "c"]), {"column_names": ["a", "b", "c"], "columns": []},
            MockRetriever(), provider, BatchSemanticConfig(max_columns_per_batch=2),
        )
        self.assertEqual([item.column_name for item in result.columns], ["a", "b", "c"])
        self.assertEqual(result.telemetry["batch_sizes"], [2, 1])
        self.assertEqual(result.telemetry["total_llm_calls"], 3)
        self.assertTrue(all("Row entity: transaction" in prompt for prompt in provider.batch_prompts))

    def test_failed_batch_retries_then_falls_back_to_unknown(self):
        provider = MockProvider([RuntimeError("bad"), RuntimeError("bad again")])
        result = analyze_batched_semantics(
            profile(["a", "b"]), {"column_names": ["a", "b"], "columns": []},
            MockRetriever(), provider, BatchSemanticConfig(max_columns_per_batch=2),
        )
        self.assertEqual([item.value_function for item in result.columns], ["unknown", "unknown"])
        self.assertEqual([item.numeric_meaning for item in result.columns], ["unknown", "unknown"])
        self.assertEqual(result.telemetry["batch_retries"], 1)
        self.assertEqual(result.telemetry["batch_failures"], 1)
        self.assertEqual(result.telemetry["total_llm_calls"], 3)
        self.assertTrue(result.warnings)

    def test_prompt_requires_domain_neutral_numeric_meaning(self):
        provider = MockProvider([[semantic("a")]])
        analyze_batched_semantics(
            profile(["a"]), {"column_names": ["a"], "columns": []},
            MockRetriever(), provider, BatchSemanticConfig(max_columns_per_batch=1),
        )
        prompt = provider.batch_prompts[0]
        for value in (
            "magnitude", "ordinal", "nominal_code", "identifier",
            "not_applicable", "unknown",
        ):
            self.assertIn(value, prompt)
        self.assertIn("arithmetic differences and central tendency", prompt)
        self.assertIn("arithmetic is meaningless", prompt)


if __name__ == "__main__":
    unittest.main()
