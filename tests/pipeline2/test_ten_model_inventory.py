import importlib
import importlib.util
import json
import unittest
from pathlib import Path

from benchmarks.tabarena.pipeline2_groundwork import EVALUATION_MODEL_IDS


ROOT = Path(__file__).resolve().parents[2]
INVENTORY_PATH = ROOT / "benchmarks" / "tabarena" / "model_inventory.json"
EXPECTED_MODELS = {
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
    "xtab",
    "tabpfn",
    "tabm",
    "tabiclv2",
    "realmlp",
    "ft_transformer",
}
REQUIRED_FIELDS = {
    "model_id",
    "canonical_model_name",
    "module_path",
    "implementation_source",
    "classification_support",
    "regression_support",
    "training_mode",
    "pretrained",
    "external_checkpoint_requirement",
    "authentication_requirement",
    "limits",
    "major_dependencies",
    "common_interface_status",
    "lifecycle_test_status",
    "benchmark_ready_status",
    "hard_limits",
    "empirical_guidance",
    "provisioning",
    "optional_diagnostics",
    "constraint_classification",
}
INTERFACE = (
    "get_model_info",
    "build_model",
    "train_model",
    "predict_model",
)


class TenModelInventoryTests(unittest.TestCase):
    def test_inventory_contains_exactly_ten_normalized_candidates(self):
        payload = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
        candidates = payload["candidates"]
        by_id = {candidate["model_id"]: candidate for candidate in candidates}

        self.assertEqual(payload["candidate_count_found"], 10)
        self.assertEqual(len(candidates), 10)
        self.assertEqual(set(by_id), EXPECTED_MODELS)
        self.assertEqual(set(EVALUATION_MODEL_IDS), EXPECTED_MODELS)
        for model_id, candidate in by_id.items():
            self.assertFalse(
                REQUIRED_FIELDS - set(candidate),
                f"{model_id} lacks normalized inventory fields",
            )
            self.assertIn(
                candidate["benchmark_ready_status"],
                {"READY", "READY_WITH_PROVISIONING", "LIMITED", "HOLD"},
            )
            self.assertEqual(candidate["common_interface_status"], "verified")
            classifications = set(candidate["constraint_classification"].values())
            self.assertLessEqual(
                classifications,
                {"HARD", "EMPIRICAL", "PROVISIONING", "NONE_VERIFIED"},
            )

    @unittest.skipUnless(
        importlib.util.find_spec("pytabkit")
        and importlib.util.find_spec("tabicl"),
        "all ten candidates require the pinned Pipeline 2 environment",
    )
    def test_all_ten_packages_expose_the_common_interface(self):
        payload = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
        for candidate in payload["candidates"]:
            package = importlib.import_module(candidate["module_path"])
            for function_name in INTERFACE:
                self.assertTrue(
                    callable(getattr(package, function_name, None)),
                    f"{candidate['model_id']} lacks {function_name}",
                )


if __name__ == "__main__":
    unittest.main()
