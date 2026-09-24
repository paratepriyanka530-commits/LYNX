import json
import unittest
from pathlib import Path

from benchmarks.tabarena.pipeline2_groundwork import EVALUATION_MODEL_IDS
from pipelines.pipeline2_prediction.build_selector_inputs import (
    build_dataset_semantics,
    validate_outputs,
    validate_performance_record,
)


ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / "pipelines" / "pipeline2_prediction" / "inputs"
TRAINING_DATA = ROOT / "pipelines" / "pipeline2_prediction" / "training_data"


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


class SelectorInputFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = _read(INPUTS / "dataset_metadata.json")
        cls.semantics = build_dataset_semantics()
        cls.targets = _read(INPUTS / "dataset_target_registry.json")
        cls.capabilities = _read(INPUTS / "model_capabilities.json")
        cls.performance = _read(TRAINING_DATA / "model_performance.json")

    def test_exact_dataset_join_and_target_counts(self):
        target_records = self.targets["datasets"]
        target_ids = [int(row["dataset_id"]) for row in target_records]
        self.assertEqual(len(target_ids), 51)
        self.assertEqual(len(set(target_ids)), 51)
        self.assertEqual([row["dataset_id"] for row in self.metadata["records"]], target_ids)
        self.assertEqual([row["dataset_id"] for row in self.semantics["records"]], target_ids)
        self.assertEqual(sum(row["task_type"] == "classification" for row in target_records), 38)
        self.assertEqual(sum(row["task_type"] == "regression" for row in target_records), 13)
        self.assertEqual(sum(row["classification_type"] == "binary" for row in target_records), 30)
        self.assertEqual(sum(row["classification_type"] == "multiclass" for row in target_records), 8)

    def test_authoritative_target_discrepancies_are_preserved(self):
        rows = {row["dataset_name"]: row for row in self.targets["datasets"]}
        self.assertEqual(
            (rows["miami_housing"]["provided_target_column"], rows["miami_housing"]["benchmark_target_column"]),
            ("structure_quality", "SALE_PRC"),
        )
        self.assertEqual(
            (rows["physiochemical_protein"]["provided_target_column"], rows["physiochemical_protein"]["benchmark_target_column"]),
            ("SpatialDistNK", "ResidualSize"),
        )

    def test_metadata_is_target_excluded_and_type_partition_is_complete(self):
        target_rows = {int(row["dataset_id"]): row for row in self.targets["datasets"]}
        for row in self.metadata["records"]:
            type_count = sum(row[f"{kind}_feature_count"] for kind in (
                "numeric", "categorical", "boolean", "datetime", "other"
            ))
            self.assertEqual(type_count, row["n_features"])
            counts = target_rows[row["dataset_id"]]["class_distribution"]
            target_count = sum(item["count"] for item in counts.values()) if counts else (
                row["n_rows"] - target_rows[row["dataset_id"]]["target_missing_count"]
            )
            self.assertEqual(target_count, row["n_rows"] - target_rows[row["dataset_id"]]["target_missing_count"])

    def test_dataset_semantic_coverage_and_schema_are_explicit(self):
        self.assertEqual(
            self.semantics["complete_count"] + self.semantics["pending_count"], 51
        )
        expected = {
            "probable_domain", "subdomain", "dataset_purpose", "row_entity",
            "target_concept", "feature_context", "semantic_summary", "domain_confidence",
        }
        for row in self.semantics["records"]:
            if row["status"] == "complete":
                features = row["semantic_meta_features"]
                self.assertEqual(set(features), expected)
                self.assertGreaterEqual(features["domain_confidence"], 0.0)
                self.assertLessEqual(features["domain_confidence"], 1.0)
                self.assertEqual(row["model"], "qwen2.5:7b")
            else:
                self.assertIsNone(row["semantic_meta_features"])

    def test_capability_roster_and_constraint_classes(self):
        records = self.capabilities["records"]
        self.assertEqual([row["model_id"] for row in records], list(EVALUATION_MODEL_IDS))
        self.assertEqual(len(records), 10)
        self.assertNotIn("cb_chl", {row["model_id"] for row in records})
        self.assertNotIn("tabpfn_crumb", {row["model_id"] for row in records})
        for row in records:
            self.assertEqual(row["empirical_guidance"]["evidence_class"], "EMPIRICAL")
            self.assertFalse(row["empirical_guidance"]["used_as_hard_gate"])
        tabpfn = next(row for row in records if row["model_id"] == "tabpfn")
        self.assertEqual(set(tabpfn["hard_limits"]["evidence_class"].values()), {"HARD"})

    def test_performance_is_an_empty_schema_not_fabricated_data(self):
        self.assertEqual(self.performance["records"], [])
        self.assertEqual(self.performance["selector_availability"], "TRAINING_OUTCOME_ONLY")
        fields = set(self.performance["record_schema"]["fields"])
        self.assertTrue({"infeasible_reason", "classification_metrics", "regression_metrics"} <= fields)

    def test_future_performance_rows_distinguish_strategies_and_outcomes(self):
        base = {
            "dataset_id": "363621", "model_id": "random_forest",
            "task_type": "classification", "feasible": True,
            "infeasible_reason": None,
            "classification_metrics": {
                "accuracy": 0.75, "balanced_accuracy": 0.75,
                "precision_macro": 0.75, "recall_macro": 0.75,
                "f1_macro": 0.75, "f2_macro": 0.75, "roc_auc": 0.8,
                "log_loss": 0.5, "mcc": 0.5,
                "confusion_matrix": [[8, 2], [3, 7]],
            },
            "regression_metrics": None,
            "primary_metric_name": "balanced_accuracy",
            "primary_metric_value": 0.75,
            "baseline_relative_score": 0.5,
            "train_runtime_seconds": 1.0,
            "inference_runtime_seconds": 0.1,
            "split_seed": 42, "source": "team_original_run", "hyperparams": {},
        }
        validate_performance_record(base)
        regression = dict(
            base, task_type="regression", classification_metrics=None,
            regression_metrics={"rmse": 1.25, "mae": 1.0, "r2": 0.5, "nrmse": 0.4},
            primary_metric_name="rmse", primary_metric_value=1.25,
        )
        validate_performance_record(regression)
        skipped = dict(
            base, feasible=False, infeasible_reason="hard feature limit",
            classification_metrics=None, primary_metric_name=None,
            primary_metric_value=None, baseline_relative_score=None,
            train_runtime_seconds=None, inference_runtime_seconds=None,
        )
        validate_performance_record(skipped)
        failed = dict(
            skipped, feasible=True,
            infeasible_reason="runtime_failure:train_or_evaluate:RuntimeError:controlled",
        )
        validate_performance_record(failed)
        with self.assertRaises(ValueError):
            validate_performance_record(dict(skipped, infeasible_reason=None))

    def test_all_artifacts_are_json_serializable(self):
        for payload in (self.metadata, self.semantics, self.targets, self.capabilities, self.performance):
            json.dumps(payload, allow_nan=False)

    def test_combined_validation_passes(self):
        result = validate_outputs({
            "metadata": self.metadata,
            "semantics": self.semantics,
            "capabilities": self.capabilities,
            "performance": self.performance,
        })
        self.assertEqual(result["base_grid_size"], 510)


if __name__ == "__main__":
    unittest.main()
