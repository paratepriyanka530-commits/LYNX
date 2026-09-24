import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from benchmarks.tabarena.pipeline2_groundwork import run_dataset_semantic_profiles
from pipelines.pipeline2_prediction.dataset_semantic_profiler import (
    MAX_REPRESENTATIVE_FEATURE_NAMES,
    Pipeline2DatasetSemanticProfile,
    build_dataset_semantic_evidence,
    representative_feature_names,
)


VALID_PROFILE = {
    "probable_domain": "commerce",
    "subdomain": "customer transactions",
    "dataset_purpose": "predict whether an account outcome occurs",
    "row_entity": "customer account observation",
    "target_concept": "account outcome",
    "feature_context": ["customer attributes", "account activity"],
    "semantic_summary": "Customer-account attributes support outcome prediction.",
    "domain_confidence": 0.8,
}


class _FakeProvider:
    calls = []
    failures_remaining = 0

    def __init__(self, model, timeout_seconds):
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.last_response_metadata = {"done_reason": "stop"}

    def generate_structured(self, prompt, response_model):
        self.__class__.calls.append((prompt, response_model))
        if self.__class__.failures_remaining:
            self.__class__.failures_remaining -= 1
            raise RuntimeError("controlled provider failure")
        return dict(VALID_PROFILE)


class DatasetSemanticProfilerTests(unittest.TestCase):
    def setUp(self):
        _FakeProvider.calls = []
        _FakeProvider.failures_remaining = 0

    def test_schema_and_deterministic_spread_cap(self):
        names = [f"feature_{index}" for index in range(1_000)]
        selected = representative_feature_names(names)
        self.assertEqual(len(selected), MAX_REPRESENTATIVE_FEATURE_NAMES)
        self.assertEqual(selected[0], names[0])
        self.assertEqual(selected[-1], names[-1])
        self.assertEqual(selected, representative_feature_names(names))
        self.assertEqual(len(set(selected)), len(selected))
        Pipeline2DatasetSemanticProfile(**VALID_PROFILE)

    def test_target_is_excluded_from_compact_evidence(self):
        evidence = build_dataset_semantic_evidence(
            dataset_name="example", task_type="classification", target_column="target",
            row_count=3, feature_count=2,
            feature_type_counts={"numeric": 1, "categorical": 1, "boolean": 0, "datetime": 0, "other": 0},
            feature_names=["a", "b"],
        )
        self.assertNotIn("target", evidence["representative_feature_names"])
        self.assertEqual(evidence["representative_feature_name_count"], 2)

    def test_runner_uses_one_request_persists_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            datasets = root / "datasets"
            output = root / "output"
            datasets.mkdir()
            pd.DataFrame({"a": [1, 2], "b": ["x", "y"], "target": [0, 1]}).to_csv(
                datasets / "example.csv", index=False
            )
            target_path = root / "targets.json"
            target_path.write_text(json.dumps({"datasets": [{
                "dataset_id": 1, "task_type": "classification",
                "benchmark_target_column": "target",
            }]}), encoding="utf-8")
            metadata_path = root / "metadata.json"
            metadata_path.write_text(json.dumps({"records": [{
                "dataset_id": 1, "n_rows": 2, "n_features": 2,
                "numeric_feature_count": 1, "categorical_feature_count": 1,
                "boolean_feature_count": 0, "datetime_feature_count": 0,
                "other_feature_count": 0,
            }]}), encoding="utf-8")
            manifest = [{
                "task_id": 1, "dataset_name": "example", "csv_file": "example.csv",
                "target_column": "target",
            }]
            with patch(
                "benchmarks.tabarena.pipeline2_groundwork.GroundworkOllamaProvider",
                _FakeProvider,
            ):
                first = run_dataset_semantic_profiles(
                    manifest, output, "qwen2.5:7b", llm_timeout=600,
                    dataset_dir=datasets, target_registry_path=target_path,
                    metadata_path=metadata_path,
                )
                second = run_dataset_semantic_profiles(
                    manifest, output, "qwen2.5:7b", llm_timeout=600,
                    dataset_dir=datasets, target_registry_path=target_path,
                    metadata_path=metadata_path,
                )
            self.assertEqual(first[0]["status"], "complete")
            self.assertEqual(second[0]["status"], "already_complete")
            self.assertEqual(len(_FakeProvider.calls), 1)
            artifact = json.loads(
                (output / "dataset_semantic_profiles" / "task_1.json").read_text(encoding="utf-8")
            )
            self.assertEqual(artifact["telemetry"]["llm_generation_requests"], 1)
            self.assertNotIn("target", artifact["evidence"]["representative_feature_names"])

    def test_one_error_is_persisted_and_does_not_stop_next_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            datasets = root / "datasets"
            output = root / "output"
            datasets.mkdir()
            manifest, targets, metadata = [], [], []
            for task_id in (1, 2):
                pd.DataFrame({"x": [1, 2], "y": [0, 1]}).to_csv(
                    datasets / f"d{task_id}.csv", index=False
                )
                manifest.append({"task_id": task_id, "dataset_name": f"d{task_id}", "csv_file": f"d{task_id}.csv", "target_column": "y"})
                targets.append({"dataset_id": task_id, "task_type": "classification", "benchmark_target_column": "y"})
                metadata.append({
                    "dataset_id": task_id, "n_rows": 2, "n_features": 1,
                    "numeric_feature_count": 1, "categorical_feature_count": 0,
                    "boolean_feature_count": 0, "datetime_feature_count": 0,
                    "other_feature_count": 0,
                })
            target_path, metadata_path = root / "targets.json", root / "metadata.json"
            target_path.write_text(json.dumps({"datasets": targets}), encoding="utf-8")
            metadata_path.write_text(json.dumps({"records": metadata}), encoding="utf-8")
            _FakeProvider.failures_remaining = 1
            with patch("benchmarks.tabarena.pipeline2_groundwork.GroundworkOllamaProvider", _FakeProvider):
                result = run_dataset_semantic_profiles(
                    manifest, output, "qwen2.5:7b", dataset_dir=datasets,
                    target_registry_path=target_path, metadata_path=metadata_path,
                )
            self.assertEqual([row["status"] for row in result], ["error", "complete"])
            self.assertEqual(len(_FakeProvider.calls), 2)
            error = json.loads((output / "dataset_semantic_profiles" / "task_1.json").read_text(encoding="utf-8"))
            self.assertEqual(error["error"]["exception_type"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()
