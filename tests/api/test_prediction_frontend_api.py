import json
import math
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

from fastapi.testclient import TestClient

from api.prediction_service import PredictionJobService
from api.run_service import ApplicationRunService
from api.server import create_app
from app.config import ApplicationSettings


def _internal_result(prediction_id, task_type="classification"):
    classification = task_type == "classification"
    predictive = (
        {
            "accuracy": 0.8,
            "balanced_accuracy": 0.72,
            "precision_macro": 0.7,
            "recall_macro": 0.72,
            "f1_macro": 0.71,
            "f2_macro": 0.715,
            "precision_weighted": 0.81,
            "recall_weighted": 0.8,
            "f1_weighted": 0.8,
            "roc_auc": 0.84,
            "log_loss": 0.42,
            "mcc": 0.51,
            "confusion_matrix": [[7, 1], [2, 6]],
            "metric_reasons": {},
        }
        if classification else
        {
            "rmse": 1.25,
            "mae": 0.8,
            "r2": 0.91,
            "nrmse": 0.14,
            "nrmse_normalizer": "population standard deviation of y_test",
            "metric_reasons": {},
        }
    )
    return {
        "schema_version": "1.0",
        "run_id": str(uuid4()),
        "prediction_id": prediction_id,
        "target_column": "outcome",
        "task_type": task_type,
        "task_resolution": {
            "status": "resolved",
            "task_type": task_type,
            "reason": "raw resolver reason must not pass through",
            "evidence": {
                "missing_count": 1,
                "semantic_reason": "SECRET_TARGET_SAMPLE",
                "unique_count": 2,
            },
        },
        "selection": {
            "selected_model": "secret_model",
            "selected_rank": 1,
            "fallback_used": False,
            "requested_model": "secret_model",
            "candidate_ranking": [{
                "rank": 1,
                "model_name": "secret_model",
                "model_family": "secret_family",
                "predicted_suitability_score": 0.99,
            }],
            "attempts": [{"error": "raw technical error"}],
        },
        "evaluation": {
            "predictive_metrics": predictive,
            "calibration_metrics": {
                "brier_score": 0.15 if classification else None,
                "ece": 0.04 if classification else None,
                "ece_n_bins": 15 if classification else None,
                "ece_method": "top_label_equal_width" if classification else None,
                "status": "available" if classification else "not_applicable",
                "reason": None if classification else "regression task",
            },
            "probability_metadata": {
                "status": "available" if classification else "not_applicable",
                "source": "SECRET_MODEL_METHOD",
                "class_order": [0, 1] if classification else None,
            },
        },
        "prediction_summary": {
            "row_count": 4,
            "probabilities_available": classification,
            "probability_class_order": [0, 1] if classification else None,
            "artifact": "predictions.csv",
        },
        "runtime": {
            "train_runtime_seconds": 1.2,
            "inference_runtime_seconds": 0.3,
            "worker_runtime_seconds": 1.8,
        },
        "orchestrator_audit": {
            "dataset": {
                "dataset_name": "safe-display.csv",
                "row_count": 20,
                "feature_names": ["SECRET_FEATURE_A", "SECRET_FEATURE_B"],
            },
            "split": {
                "split_seed": 42,
                "test_size": 0.2,
                "stratified": classification,
                "train_row_count": 16,
                "test_row_count": 4,
            },
            "selector": {
                "representation": "D",
                "semantic_profile": {"semantic_summary": "SECRET_SEMANTICS"},
                "target_info": {"n_classes": 2 if classification else None},
            },
            "artifacts": {
                "model_reference": "secret.module.Model",
                "preprocessor_reference": "secret.module.Preprocessor",
                "target_encoder_reference": "sklearn.LabelEncoder",
                "transformed_feature_names": ["SECRET_TRANSFORMED"],
            },
            "warnings": ["SECRET_WARNING"],
        },
    }


class PredictionFrontendApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.run_service = ApplicationRunService(ApplicationSettings(
            runtime_root=Path(self.temporary.name) / "runtime",
            prediction_worker_count=1,
        ))
        self.prediction_service = PredictionJobService(
            self.run_service, auto_execute=False
        )
        self.client = TestClient(create_app(
            service=self.run_service,
            prediction_service=self.prediction_service,
        ))

    def tearDown(self):
        self.prediction_service.shutdown(wait=True)
        self.run_service.shutdown(wait=True)
        self.temporary.cleanup()

    def new_prediction(self, task_type="classification", rows=None):
        run = self.run_service.store.create("data.csv", ".csv")
        run_id = run["application_run_id"]
        state = self.prediction_service.store.create(run_id, "outcome")
        prediction_id = state["prediction_id"]
        internal = _internal_result(prediction_id, task_type)
        self.prediction_service.store.write_result(run_id, prediction_id, internal)
        if rows is None:
            rows = (
                [
                    ["No", "No", 0.8, 0.2],
                    ["Yes", "No", 0.6, 0.4],
                    ["Yes", "Yes", 0.1, 0.9],
                    ["No", "No", 0.7, 0.3],
                ]
                if task_type == "classification" else
                [[1.0, 1.5], [2.0, 1.0], [3.0, 3.25], [4.0, 4.0]]
            )
        fields = (
            ["actual", "prediction", "probability_0", "probability_1"]
            if task_type == "classification" else ["actual", "prediction"]
        )
        self.prediction_service.store.write_prediction_rows(
            run_id, prediction_id, fields, rows
        )
        self.prediction_service.store.update(
            run_id, prediction_id, state="complete", stage="complete"
        )
        return run_id, prediction_id, internal

    @staticmethod
    def _all_keys(value):
        keys = set()
        if isinstance(value, dict):
            for key, child in value.items():
                keys.add(key)
                keys.update(PredictionFrontendApiTests._all_keys(child))
        elif isinstance(value, list):
            for child in value:
                keys.update(PredictionFrontendApiTests._all_keys(child))
        return keys

    def test_classification_safe_result_projection_and_internal_artifact_unchanged(self):
        run_id, prediction_id, internal = self.new_prediction()
        path = self.prediction_service.store._paths(run_id, prediction_id)["result"]
        before = path.read_bytes()
        response = self.client.get(
            f"/api/runs/{run_id}/predictions/{prediction_id}/result"
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["status"], "complete")
        self.assertEqual(payload["task_type"], "classification")
        self.assertEqual(payload["target"], "outcome")
        self.assertEqual(payload["dataset"], {
            "name": "safe-display.csv", "total_rows": 20,
            "training_rows": 16, "test_rows": 4, "feature_count": 2,
            "target_missing_count": 1,
        })
        self.assertEqual(payload["metrics"]["primary"]["balanced_accuracy"], 0.72)
        self.assertEqual(payload["metrics"]["confusion_matrix"], [[7, 1], [2, 6]])
        self.assertEqual(payload["task_resolution"]["explanation"], "Resolved as a classification target.")
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(json.loads(before), internal)

    def test_regression_safe_result_projection(self):
        run_id, prediction_id, _internal = self.new_prediction("regression")
        payload = self.client.get(
            f"/api/runs/{run_id}/predictions/{prediction_id}/result"
        ).json()
        self.assertEqual(payload["task_type"], "regression")
        self.assertEqual(payload["metrics"], {
            "rmse": 1.25, "mae": 0.8, "r2": 0.91, "nrmse": 0.14,
            "nrmse_normalizer": "population standard deviation of y_test",
        })
        self.assertFalse(payload["predictions"]["has_probabilities"])

    def test_internal_selector_resolver_references_features_and_worker_data_absent(self):
        run_id, prediction_id, _internal = self.new_prediction()
        payload = self.client.get(
            f"/api/runs/{run_id}/predictions/{prediction_id}/result"
        ).json()
        keys = self._all_keys(payload)
        forbidden_keys = {
            "selection", "selected_model", "selected_rank", "requested_model",
            "candidate_ranking", "fallback_used", "attempts", "orchestrator_audit",
            "evidence", "semantic_reason", "feature_names", "transformed_feature_names",
            "worker_runtime_seconds", "model_reference", "preprocessor_reference",
            "target_encoder_reference", "split_seed",
        }
        self.assertTrue(keys.isdisjoint(forbidden_keys), keys & forbidden_keys)
        serialized = json.dumps(payload)
        for forbidden_value in (
            "SECRET_TARGET_SAMPLE", "SECRET_FEATURE", "SECRET_SEMANTICS",
            "secret.module", "SECRET_WARNING", "raw resolver reason",
        ):
            self.assertNotIn(forbidden_value, serialized)

    def test_classification_pagination_confidence_and_no_probability_columns(self):
        run_id, prediction_id, _internal = self.new_prediction()
        response = self.client.get(
            f"/api/runs/{run_id}/predictions/{prediction_id}/rows?offset=1&limit=2"
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(
            {key: payload[key] for key in ("offset", "limit", "returned", "total")},
            {"offset": 1, "limit": 2, "returned": 2, "total": 4},
        )
        self.assertEqual(payload["rows"][0], {
            "actual": "Yes", "prediction": "No", "confidence": 0.6,
        })
        for row in payload["rows"]:
            self.assertEqual(set(row), {"actual", "prediction", "confidence"})
            self.assertNotIn("row_id", row)

    def test_regression_pagination_and_absolute_error(self):
        run_id, prediction_id, _internal = self.new_prediction("regression")
        payload = self.client.get(
            f"/api/runs/{run_id}/predictions/{prediction_id}/rows?limit=3"
        ).json()
        self.assertEqual(payload["returned"], 3)
        self.assertEqual(payload["rows"][0], {
            "actual": 1.0, "prediction": 1.5, "absolute_error": 0.5,
        })
        self.assertEqual(set(payload["rows"][0]), {
            "actual", "prediction", "absolute_error",
        })

    def test_pagination_defaults_hard_limit_and_invalid_values(self):
        run_id, prediction_id, _internal = self.new_prediction()
        base = f"/api/runs/{run_id}/predictions/{prediction_id}/rows"
        default = self.client.get(base)
        self.assertEqual(default.status_code, 200)
        self.assertEqual(default.json()["limit"], 50)
        for query in ("limit=201", "limit=0", "offset=-1", "offset=no", "limit=1.5"):
            response = self.client.get(f"{base}?{query}")
            self.assertEqual(response.status_code, 422, response.text)
            self.assertEqual(response.json()["error"]["code"], "invalid_pagination")

    def test_cross_run_access_is_blocked(self):
        first_run, prediction_id, _internal = self.new_prediction()
        second = self.run_service.store.create("other.csv", ".csv")
        second_run = second["application_run_id"]
        response = self.client.get(
            f"/api/runs/{second_run}/predictions/{prediction_id}/rows"
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "prediction_not_found")
        self.assertNotEqual(first_run, second_run)

    def test_malformed_and_unknown_ids_are_controlled(self):
        run_id, prediction_id, _internal = self.new_prediction()
        malformed_prediction = self.client.get(
            f"/api/runs/{run_id}/predictions/not-a-uuid/rows"
        )
        malformed_run = self.client.get(
            f"/api/runs/not-a-uuid/predictions/{prediction_id}/rows"
        )
        unknown_prediction = self.client.get(
            f"/api/runs/{run_id}/predictions/{uuid4()}/rows"
        )
        self.assertEqual(
            (malformed_prediction.status_code, malformed_prediction.json()["error"]["code"]),
            (400, "malformed_prediction_id"),
        )
        self.assertEqual(
            (malformed_run.status_code, malformed_run.json()["error"]["code"]),
            (400, "malformed_run_id"),
        )
        self.assertEqual(
            (unknown_prediction.status_code, unknown_prediction.json()["error"]["code"]),
            (404, "prediction_not_found"),
        )

    def test_queued_running_and_failed_rows_are_controlled(self):
        run = self.run_service.store.create("states.csv", ".csv")
        run_id = run["application_run_id"]
        expected = {
            "queued": "prediction_not_complete",
            "running": "prediction_not_complete",
            "failed": "prediction_failed",
        }
        for state, error_code in expected.items():
            job = self.prediction_service.store.create(run_id, "outcome")
            if state != "queued":
                self.prediction_service.store.update(
                    run_id, job["prediction_id"], state=state,
                    error=({"code": "safe", "message": "safe"} if state == "failed" else None),
                )
            response = self.client.get(
                f"/api/runs/{run_id}/predictions/{job['prediction_id']}/rows"
            )
            self.assertEqual(response.status_code, 409)
            self.assertEqual(response.json()["error"]["code"], error_code)

    def test_missing_malformed_and_unexpected_csv_are_controlled(self):
        for mode in ("missing", "malformed", "schema"):
            run_id, prediction_id, _internal = self.new_prediction()
            path = self.prediction_service.store._paths(run_id, prediction_id)["predictions"]
            if mode == "missing":
                path.unlink()
                expected = "prediction_rows_unavailable"
            elif mode == "malformed":
                path.write_bytes(b"actual,prediction\n\xff,yes\n")
                expected = "prediction_rows_invalid"
            else:
                path.write_text("actual,prediction,secret_feature\nno,no,secret\n", encoding="utf-8")
                expected = "prediction_rows_invalid"
            response = self.client.get(
                f"/api/runs/{run_id}/predictions/{prediction_id}/rows"
            )
            self.assertEqual(response.status_code, 500, response.text)
            self.assertEqual(response.json()["error"]["code"], expected)
            self.assertNotIn(str(path), response.text)

    def test_non_finite_values_become_null(self):
        run_id, prediction_id, _internal = self.new_prediction(rows=[
            ["No", "No", float("nan"), 0.2],
            ["Yes", "Yes", float("inf"), 0.9],
        ])
        rows = self.client.get(
            f"/api/runs/{run_id}/predictions/{prediction_id}/rows"
        ).json()["rows"]
        self.assertIsNone(rows[0]["confidence"])
        self.assertIsNone(rows[1]["confidence"])

        run_id, prediction_id, _internal = self.new_prediction(
            "regression", rows=[[float("nan"), 1.0], [2.0, float("inf")]]
        )
        rows = self.client.get(
            f"/api/runs/{run_id}/predictions/{prediction_id}/rows"
        ).json()["rows"]
        self.assertEqual(rows[0], {
            "actual": None, "prediction": 1.0, "absolute_error": None,
        })
        self.assertEqual(rows[1], {
            "actual": 2.0, "prediction": None, "absolute_error": None,
        })

    def test_missing_and_malformed_result_are_controlled(self):
        run = self.run_service.store.create("bad-result.csv", ".csv")
        run_id = run["application_run_id"]
        for mode in ("missing", "malformed"):
            job = self.prediction_service.store.create(run_id, "outcome")
            prediction_id = job["prediction_id"]
            path = self.prediction_service.store._paths(run_id, prediction_id)["result"]
            if mode == "malformed":
                path.write_text("{not-json", encoding="utf-8")
            self.prediction_service.store.update(
                run_id, prediction_id, state="complete", stage="complete"
            )
            response = self.client.get(
                f"/api/runs/{run_id}/predictions/{prediction_id}/result"
            )
            self.assertEqual(response.status_code, 500)
            self.assertEqual(
                response.json()["error"]["code"], "prediction_result_unavailable"
            )


if __name__ == "__main__":
    unittest.main()
