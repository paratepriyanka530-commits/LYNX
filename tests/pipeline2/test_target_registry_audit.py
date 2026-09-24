import csv
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from benchmarks.tabarena.build_target_registry import (
    _imbalance_category,
    build_target_registry,
)


ROOT = Path(__file__).resolve().parents[2]


class TargetRegistryAuditTests(unittest.TestCase):
    def test_threshold_boundaries_are_exact(self):
        self.assertEqual(_imbalance_category(0.009), "EXTREME_IMBALANCE")
        self.assertEqual(_imbalance_category(0.01), "SEVERE_IMBALANCE")
        self.assertEqual(_imbalance_category(0.049), "SEVERE_IMBALANCE")
        self.assertEqual(_imbalance_category(0.05), "MODERATE_IMBALANCE")
        self.assertEqual(_imbalance_category(0.199), "MODERATE_IMBALANCE")
        self.assertEqual(_imbalance_category(0.20), "NOT_IMBALANCED")

    def test_builder_uses_exact_targets_and_excludes_missing_from_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            datasets = root / "datasets"
            datasets.mkdir()
            pd.DataFrame({
                "x": range(5),
                "label": ["pdf", "pdf", "pdf", "pdf", "pdf"],
                "upstream_label": ["a", "a", "a", "b", None],
            }).to_csv(
                datasets / "classification.csv", index=False
            )
            pd.DataFrame({"x": range(3), "value": [1.0, 2.0, None]}).to_csv(
                datasets / "regression.csv", index=False
            )
            pd.DataFrame({"Target": [0, 1]}).to_csv(
                datasets / "case-sensitive.csv", index=False
            )
            source = {
                "datasets": [
                    {"dataset_name": "classification", "dataset_file": "classification.csv", "target_column": "label"},
                    {"dataset_name": "regression", "dataset_file": "regression.csv", "target_column": "value"},
                    {"dataset_name": "case-sensitive", "dataset_file": "case-sensitive.csv", "target_column": "target"},
                ]
            }
            manifest = {
                "datasets": [
                    {"task_id": 1, "dataset_name": "classification", "csv_file": "classification.csv", "target_column": "upstream_label", "task_type": "Supervised Classification"},
                    {"task_id": 2, "dataset_name": "regression", "csv_file": "regression.csv", "target_column": "value", "task_type": "Supervised Regression"},
                    {"task_id": 3, "dataset_name": "case-sensitive", "csv_file": "case-sensitive.csv", "target_column": "target", "task_type": "Supervised Classification"},
                ]
            }
            source_path = root / "source.json"
            manifest_path = root / "manifest.json"
            source_path.write_text(json.dumps(source), encoding="utf-8")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            payload = build_target_registry(
                source_path,
                manifest_path,
                datasets,
                root / "inputs" / "dataset_target_registry.json",
                root / "artifacts",
                expected_count=3,
            )

            by_name = {item["dataset_name"]: item for item in payload["datasets"]}
            classification = by_name["classification"]
            self.assertEqual(classification["status"], "resolved")
            self.assertEqual(classification["provided_target_column"], "label")
            self.assertEqual(classification["authoritative_target_column"], "upstream_label")
            self.assertEqual(classification["benchmark_target_column"], "upstream_label")
            self.assertEqual(classification["target_column"], "upstream_label")
            self.assertEqual(
                classification["target_resolution_status"],
                "resolved_to_authoritative_target",
            )
            self.assertEqual(classification["target_missing_count"], 1)
            self.assertEqual(
                sum(item["count"] for item in classification["class_distribution"].values()),
                4,
            )
            self.assertAlmostEqual(
                sum(item["proportion"] for item in classification["class_distribution"].values()),
                1.0,
            )
            self.assertEqual(classification["minority_class_ratio"], 0.25)
            self.assertFalse(classification["cb_chl_eligible"])

            regression = by_name["regression"]
            self.assertEqual(regression["target_missing_count"], 1)
            self.assertAlmostEqual(regression["target_missing_pct"], 100 / 3)
            for field in (
                "classification_type", "n_classes", "class_distribution",
                "minority_class_ratio", "imbalance_category",
            ):
                self.assertIsNone(regression[field])
            self.assertFalse(regression["cb_chl_eligible"])

            unresolved = by_name["case-sensitive"]
            self.assertEqual(unresolved["status"], "unresolved")
            self.assertIn("absent from the exact CSV header", unresolved["unresolved_reason"])
            self.assertTrue((root / "artifacts" / "imbalance_audit.json").is_file())
            with (root / "artifacts" / "imbalance_audit.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 3)

    def test_generated_51_dataset_artifacts_are_complete(self):
        input_path = (
            ROOT / "pipelines" / "pipeline2_prediction" / "inputs"
            / "dataset_target_registry.json"
        )
        artifact_path = (
            ROOT / "benchmarks" / "tabarena" / "pipeline2_artifacts"
            / "dataset_target_registry.json"
        )
        audit_path = artifact_path.with_name("imbalance_audit.json")
        input_payload = json.loads(input_path.read_text(encoding="utf-8"))
        artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))

        self.assertEqual(input_payload, artifact_payload)
        self.assertEqual(len(artifact_payload["datasets"]), 51)
        self.assertTrue(artifact_payload["validation"]["passed"])
        self.assertEqual(len(audit_payload["records"]), 51)
        self.assertTrue(all(
            item["status"] == "resolved" for item in artifact_payload["datasets"]
        ))

        classification = [
            item for item in artifact_payload["datasets"]
            if item["task_type"] == "classification"
        ]
        regression = [
            item for item in artifact_payload["datasets"]
            if item["task_type"] == "regression"
        ]
        self.assertEqual((len(classification), len(regression)), (38, 13))
        by_name = {item["dataset_name"]: item for item in artifact_payload["datasets"]}
        self.assertEqual(
            by_name["miami_housing"]["provided_target_column"],
            "structure_quality",
        )
        self.assertEqual(
            by_name["miami_housing"]["benchmark_target_column"], "SALE_PRC"
        )
        self.assertEqual(
            by_name["physiochemical_protein"]["provided_target_column"],
            "SpatialDistNK",
        )
        self.assertEqual(
            by_name["physiochemical_protein"]["benchmark_target_column"],
            "ResidualSize",
        )
        for item in classification:
            distribution = item["class_distribution"]
            self.assertAlmostEqual(
                sum(part["proportion"] for part in distribution.values()), 1.0
            )
            self.assertEqual(
                sum(part["count"] for part in distribution.values()),
                item["target_row_count"] - item["target_missing_count"],
            )
            self.assertIsNotNone(item["minority_class_ratio"])
        for item in regression:
            self.assertIsNone(item["class_distribution"])
            self.assertIsNone(item["minority_class_ratio"])
            self.assertFalse(item["cb_chl_eligible"])
        eligible = {
            item["dataset_name"] for item in artifact_payload["datasets"]
            if item["cb_chl_eligible"]
        }
        self.assertEqual(eligible, {"anneal", "MIC"})


if __name__ == "__main__":
    unittest.main()
