import unittest

from pipelines.pipeline1_cleaning.semantics.column_understanding import understand_column
from pipelines.pipeline1_cleaning.semantics.semantic_schema import DatasetSemanticResult


class EmptyRetriever:
    def retrieve(self, query, top_k, categories):
        return []


class ContractCheckingLLM:
    def generate_structured(self, prompt, response_model):
        self.required_fields = set(response_model.model_json_schema()["required"])
        self.prompt = prompt
        return {
            "semantic_type": "categorical",
            "value_function": "classification",
            "concept_description": "Values denote a reusable coded vocabulary.",
            "confidence": 0.85,
            "reasoning_summary": "Samples repeat and behave as discrete labels.",
        }


class ColumnSemanticContractTests(unittest.TestCase):
    def test_required_contract_is_produced_before_role_resolution(self):
        llm = ContractCheckingLLM()
        result = understand_column(
            column_evidence={
                "column_name": "class_code",
                "physical_dtype": "object",
                "missing": {"percentage": 0.0},
                "uniqueness": {"unique_ratio": 0.1},
                "repetition": {"repeated_occurrences": 9},
                "sample_values": ["A", "B"],
            },
            dataset_semantics=DatasetSemanticResult(
                row_entity="observation", confidence=0.8
            ),
            retriever=EmptyRetriever(),
            llm_provider=llm,
        )

        expected_fields = {
            "semantic_type", "value_function", "concept_description",
            "confidence", "reasoning_summary",
        }
        self.assertTrue(expected_fields.issubset(llm.required_fields))
        self.assertIn("semantic value function, not structural role", llm.prompt)
        self.assertEqual(result.semantic_type, "categorical")
        self.assertEqual(result.value_function, "classification")
        self.assertEqual(result.semantic_role, "unknown")


if __name__ == "__main__":
    unittest.main()
