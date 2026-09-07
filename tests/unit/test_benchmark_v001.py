import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.pipeline1_semantics.run_benchmark_v001 import (
    CallTimingRecorder,
    TimedLLMProvider,
    TimedRetriever,
    discover_datasets,
    run_benchmark,
    run_consistency_checks,
    timed_semantic_operation,
)


class FakeFrame:
    shape = (3, 2)


class FakeRetriever:
    def retrieve(self, **kwargs):
        return []


class FakeLLM:
    def generate_structured(self, **kwargs):
        return {"result": "ok"}


def fake_runtime_factory(model_name):
    profiles = [
        {
            "column_name": "record_key",
            "physical_dtype": "object",
            "missing": {"count": 0, "percentage": 0.0},
            "uniqueness": {"unique_count": 3, "unique_ratio": 1.0},
            "repetition": {
                "repeated_unique_count": 0,
                "repeated_occurrences": 0,
                "repetition_rate": 0.0,
            },
            "sample_values": ["R1", "R2", "R3"],
        },
        {
            "column_name": "broken_column",
            "physical_dtype": "object",
            "missing": {"count": 0, "percentage": 0.0},
            "uniqueness": {"unique_count": 2, "unique_ratio": 0.667},
            "repetition": {
                "repeated_unique_count": 1,
                "repeated_occurrences": 2,
                "repetition_rate": 0.667,
            },
            "sample_values": ["A", "B"],
        },
    ]

    def profile_dataset(dataframe):
        return {
            "shape": {"rows": 3, "columns": 2},
            "column_names": ["record_key", "broken_column"],
            "dtype_distribution": {"object": 2},
            "missingness": {"total_missing_cells": 0,
                            "overall_missing_percentage": 0.0},
            "duplicates": {"exact_duplicate_rows": 0, "percentage": 0.0},
            "column_profiles": profiles,
        }

    def build_column_evidence(dataset_profile, column_profile):
        return {"column": column_profile, "dataset_context": {}}

    def understand_column(column_evidence, **kwargs):
        if column_evidence["column_name"] == "broken_column":
            raise RuntimeError("simulated semantic failure")
        return {
            "column_name": "record_key",
            "semantic_type": "identifier",
            "semantic_role": "unknown",
            "value_function": "entity_reference",
            "concept_description": "record",
            "confidence": 0.9,
            "reasoning_summary": "Values identify records.",
            "rag_sources": [],
        }

    return {
        "load_dataset": lambda path: {"dataframe": FakeFrame()},
        "profile_dataset": profile_dataset,
        "build_dataset_evidence": lambda profile: {"column_names": profile["column_names"]},
        "build_column_evidence": build_column_evidence,
        "understand_dataset": lambda **kwargs: {
            "dataset_domain": "generic",
            "dataset_description": "Synthetic smoke dataset.",
            "row_entity": "record",
            "confidence": 0.9,
            "reasoning_summary": "One row represents one record.",
            "rag_sources": [],
        },
        "understand_column": understand_column,
        "resolve_column_role": lambda **kwargs: {
            "column_name": "record_key",
            "referenced_entity": "record",
            "entity_alignment": "same_as_row_entity",
            "semantic_role": "primary_identifier",
            "confidence": 0.9,
            "reasoning_summary": "Record aligns with row entity record.",
            "rag_sources": [],
        },
        "retriever": FakeRetriever(),
        "llm_provider": object(),
        "indexed_chunks": 10,
        "embedding_model": "fake-smoke-embedding",
    }


class BenchmarkV001Tests(unittest.TestCase):
    def test_timing_wrappers_attribute_rag_and_llm_calls(self):
        recorder = CallTimingRecorder()
        retriever = TimedRetriever(FakeRetriever(), recorder)
        llm = TimedLLMProvider(FakeLLM(), recorder)

        def bundled_operation(retriever, llm_provider):
            retriever.retrieve(query="semantic", top_k=1)
            retriever.retrieve(query="pattern", top_k=1)
            return llm_provider.generate_structured(prompt="test")

        result, timings, counts, llm_metrics, total = timed_semantic_operation(
            bundled_operation,
            recorder,
            ["semantic_context_rag_retrieval", "value_pattern_rag_retrieval"],
            "column_understanding_llm_inference",
            retriever=retriever,
            llm_provider=llm,
        )

        self.assertEqual(result, {"result": "ok"})
        self.assertEqual(counts["semantic_context_rag_retrieval"], 1)
        self.assertEqual(counts["value_pattern_rag_retrieval"], 1)
        self.assertEqual(counts["column_understanding_llm_inference"], 1)
        self.assertEqual(
            llm_metrics["column_understanding_llm_inference"][0][
                "prompt_character_length"
            ],
            4,
        )
        self.assertGreaterEqual(total, sum(timings.values()))

    def test_consistency_checks_are_diagnostic_and_non_mutating(self):
        column_semantics = {
            "semantic_type": "unknown",
            "value_function": "entity_reference",
            "confidence": 0.95,
        }
        role_resolution = {
            "referenced_entity": "participant",
            "entity_alignment": "same_as_row_entity",
            "semantic_role": "foreign_identifier",
            "confidence": 0.9,
        }
        original_column = dict(column_semantics)
        original_role = dict(role_resolution)

        checks = run_consistency_checks(column_semantics, role_resolution)
        rules = {check["rule"] for check in checks}

        self.assertIn("same_entity_foreign_role", rules)
        self.assertIn("entity_reference_unknown_type", rules)
        self.assertIn("high_confidence_unknown_semantics", rules)
        self.assertEqual(column_semantics, original_column)
        self.assertEqual(role_resolution, original_role)

    def test_discovery_records_supported_and_skipped_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "sample.csv").write_text("a\n1\n", encoding="utf-8")
            (directory / "notes.txt").write_text("control", encoding="utf-8")

            datasets, discovered, skipped = discover_datasets(directory)

            self.assertEqual([path.name for path in datasets], ["sample.csv"])
            self.assertEqual(len(discovered), 2)
            self.assertEqual(skipped[0]["name"], "notes.txt")
            self.assertIn("unsupported extension", skipped[0]["reason"])

    def test_smoke_outputs_parse_and_column_failure_is_isolated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_directory = root / "datasets"
            output_directory = root / "benchmark_v001_baseline"
            dataset_directory.mkdir()
            (dataset_directory / "small.csv").write_text(
                "record_key,broken_column\nR1,A\nR2,A\nR3,B\n",
                encoding="utf-8",
            )

            exit_code = run_benchmark(
                dataset_directory=dataset_directory,
                output_directory=output_directory,
                runtime_factory=fake_runtime_factory,
            )

            self.assertEqual(exit_code, 1)
            expected_files = {
                "manifest.json", "datasets.jsonl", "columns.jsonl",
                "errors.jsonl", "summary.json",
            }
            self.assertEqual(
                {path.name for path in output_directory.iterdir()}, expected_files
            )

            manifest = json.loads(
                (output_directory / "manifest.json").read_text(encoding="utf-8")
            )
            summary = json.loads(
                (output_directory / "summary.json").read_text(encoding="utf-8")
            )
            dataset_records = [
                json.loads(line) for line in
                (output_directory / "datasets.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            column_records = [
                json.loads(line) for line in
                (output_directory / "columns.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            error_records = [
                json.loads(line) for line in
                (output_directory / "errors.jsonl").read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(manifest["status"], "completed_with_failures")
            self.assertEqual(summary["run"]["columns_attempted"], 2)
            self.assertEqual(summary["run"]["columns_successful"], 1)
            self.assertEqual(summary["run"]["columns_failed"], 1)
            self.assertEqual(dataset_records[0]["status"], "partial")
            self.assertEqual(len(column_records), 1)
            self.assertEqual(error_records[0]["column_name"], "broken_column")
            self.assertEqual(error_records[0]["stage"], "column_understanding")


if __name__ == "__main__":
    unittest.main()
