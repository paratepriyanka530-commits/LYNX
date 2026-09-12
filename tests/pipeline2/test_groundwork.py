import unittest

import pandas as pd

from benchmarks.tabarena.pipeline2_groundwork import (
    MODEL_CAPABILITIES,
    compute_model_feasibility,
    feasibility,
    statistical_meta_features,
    task_type_supported,
    validate_log_record,
)


class GroundworkTests(unittest.TestCase):
    def test_statistical_features_include_target_imbalance(self):
        frame = pd.DataFrame({"x": [1.0, 2.0, None, 4.0], "kind": ["a", "a", "b", "b"], "y": [0, 0, 0, 1]})
        result = statistical_meta_features(frame, "y")
        self.assertEqual(result["row_count"], 4)
        self.assertEqual(result["feature_count"], 2)
        self.assertEqual(result["target_class_count"], 2)
        self.assertEqual(result["target_majority_ratio"], 0.75)

    def test_foundation_model_feasibility_gate(self):
        self.assertEqual(
            feasibility(
                "TabPFN", 10_001, 10, "Supervised Classification", 2
            )[0],
            False,
        )
        self.assertEqual(
            feasibility(
                "TabPFN", 10_000, 501, "Supervised Classification", 2
            )[0],
            False,
        )
        self.assertEqual(
            feasibility(
                "TabPFN", 10_000, 500, "Supervised Classification", 11
            )[0],
            False,
        )
        self.assertEqual(
            feasibility(
                "TabPFN", 10_000, 500, "Supervised Classification", 10
            )[0],
            True,
        )

    def test_model_task_type_capabilities_match_implemented_wrappers(self):
        self.assertEqual(
            MODEL_CAPABILITIES["randomforest"]["task_type_supported"],
            ["classification", "regression"],
        )
        for model_id in ("lightgbm", "xgboost", "catboost", "tabpfn"):
            self.assertEqual(
                MODEL_CAPABILITIES[model_id]["task_type_supported"],
                ["classification"],
            )
            self.assertFalse(
                task_type_supported(model_id, "Supervised Regression")
            )
            self.assertEqual(
                feasibility(
                    model_id, 100, 4, task_type="Supervised Regression"
                )[0],
                False,
            )
        self.assertTrue(
            task_type_supported("random_forest", "Supervised Regression")
        )
        self.assertTrue(task_type_supported("xtab", "Supervised Regression"))
        self.assertTrue(task_type_supported("realmlp", "Supervised Regression"))
        self.assertTrue(task_type_supported("tabm", "Supervised Regression"))

    def test_all_models_reject_unsupported_tasks(self):
        for model_id in ("lightgbm", "xgboost", "catboost", "tabpfn"):
            self.assertFalse(task_type_supported(model_id, "regression"))
            self.assertFalse(
                feasibility(model_id, 100, 4, "regression", 0)[0]
            )
        for model_id in (
            "random_forest", "xtab", "realmlp", "tabm", "tabiclv2",
            "ft_transformer",
        ):
            self.assertTrue(task_type_supported(model_id, "classification"))
            self.assertTrue(task_type_supported(model_id, "regression"))

    def test_xtab_ten_row_minimum(self):
        self.assertFalse(feasibility("xtab", 9, 4, "classification", 2)[0])
        self.assertTrue(feasibility("xtab", 10, 4, "classification", 2)[0])
        self.assertFalse(feasibility("xtab", 9, 4, "regression")[0])
        self.assertTrue(feasibility("xtab", 10, 4, "regression")[0])

    def test_binary_only_implementations_reject_multiclass(self):
        for model_id in ("lightgbm", "xgboost", "catboost"):
            feasible, reason = feasibility(
                model_id, 100, 4, "classification", class_count=3
            )
            self.assertFalse(feasible)
            self.assertIn("class_count=3", reason)
        for model_id in (
            "random_forest", "xtab", "tabpfn", "realmlp", "tabm",
            "tabiclv2", "ft_transformer",
        ):
            self.assertTrue(
                feasibility(
                    model_id, 100, 4, "classification", class_count=3
                )[0]
            )

    def test_unknown_class_count_stays_unresolved_when_limit_exists(self):
        for model_id in ("lightgbm", "xgboost", "catboost", "tabpfn"):
            feasible, reason = feasibility(
                model_id, 100, 4, "classification", class_count=None
            )
            self.assertIsNone(feasible)
            self.assertIn("Class count", reason)

    def test_feasibility_artifact_marks_regression_models_before_run(self):
        import tempfile
        from pathlib import Path

        manifest = [{
            "task_id": 1,
            "dataset_name": "regression-example",
            "task_type": "Supervised Regression",
        }]
        statistics = [{
            "dataset_id": "1",
            "statistical_meta_features": {"row_count": 100, "feature_count": 4},
        }]
        with tempfile.TemporaryDirectory() as directory:
            records = compute_model_feasibility(
                manifest, statistics, Path(directory)
            )
        by_model = {record["model_id"]: record for record in records}
        self.assertTrue(by_model["random_forest"]["feasible"])
        self.assertTrue(by_model["xtab"]["feasible"])
        for model_id in ("lightgbm", "xgboost", "catboost", "tabpfn"):
            self.assertFalse(by_model[model_id]["task_type_supported"])
            self.assertFalse(by_model[model_id]["feasible"])

    def test_infeasible_score_is_rejected(self):
        record = {
            "dataset_id": "1", "model_id": "TabICL", "target_column": "y",
            "task_type": "classification", "feasible": False,
            "status": "infeasible_skip", "outcome_reason": "capacity",
            "metric_name": "roc_auc", "metric_value": 0.9,
            "baseline_relative_score": None, "train_runtime_seconds": None,
            "inference_runtime_seconds": None, "runtime_seconds": None,
            "split_seed": 42, "hyperparams": {}, "semantic_meta_features": {},
            "statistical_meta_features": {}, "source": "team-original-run",
        }
        with self.assertRaisesRegex(ValueError, "must not contain"):
            validate_log_record(record)


if __name__ == "__main__":
    unittest.main()
