import hashlib
import json
from pathlib import Path
from threading import Event, Lock
import tempfile
import time
import unittest

from fastapi.testclient import TestClient

from api.prediction_service import PredictionJobService, PredictionStore
from api.run_service import ApplicationRunService
from api.server import create_app
from app.config import ApplicationSettings
from tests.api.test_application_flow import _write_successful_pipeline1_artifacts


def _resolution(task_type="classification", status="resolved"):
    return {
        "status": status,
        "task_type": task_type if status == "resolved" else None,
        "reason": "fixture evidence",
        "evidence": {"non_null_count": 8, "unique_count": 2},
    }


def _orchestration(task_type="classification", *, fallback=False, metric=0.75):
    attempts = []
    if fallback:
        attempts.append({
            "rank": 1, "model_id": "xgboost", "status": "failed",
            "failure_stage": "model_fit", "error": "SECRET raw exception",
            "train_runtime_seconds": 0.1, "inference_runtime_seconds": None,
        })
    selected_rank = 2 if fallback else 1
    selected_model = "lightgbm" if fallback else "xgboost"
    attempts.append({
        "rank": selected_rank, "model_id": selected_model, "status": "success",
        "failure_stage": None, "error": None,
        "train_runtime_seconds": 0.2, "inference_runtime_seconds": 0.03,
    })
    predictions = {
        "y_true": [0, 1], "y_pred": [0, 1],
        "probabilities": [[0.8, 0.2], [0.1, 0.9]] if task_type == "classification" else None,
        "probability_class_order": [0, 1] if task_type == "classification" else None,
        "decoded_labels": None,
    }
    predictive_metrics = (
        {
            "accuracy": metric,
            "balanced_accuracy": metric,
            "precision_macro": metric,
            "recall_macro": metric,
            "f1_macro": metric,
            "roc_auc": metric,
            "log_loss": 1.0 - metric,
            "mcc": metric,
            "confusion_matrix": [[1, 0], [0, 1]],
        }
        if task_type == "classification" else
        {
            "rmse": 1.0, "mae": 0.75, "r2": metric, "nrmse": 0.2,
            "nrmse_normalizer": "population standard deviation of y_test",
        }
    )
    return {
        "schema_version": "1.0", "run_id": "internal-orchestrator-id",
        "dataset": {"dataset_name": "data.csv", "row_count": 8, "feature_names": ["feature"]},
        "split": {"split_seed": 42, "test_size": 0.2, "train_indices": [0, 1, 2, 3, 4, 5], "test_indices": [6, 7], "stratified": task_type == "classification"},
        "selector": {
            "artifact_path": "C:/SECRET/model.joblib", "artifact_sha256": "SECRET_HASH",
            "representation": "combined", "semantic_profile": {"probable_domain": "unknown"},
            "target_info": {"n_classes": 2},
            "ranking": [{"rank": index, "model_name": name, "score": 1 / index} for index, name in enumerate([
                "xgboost", "lightgbm", "catboost", "random_forest", "realmlp",
                "tabiclv2", "tabm", "tabpfn", "xtab", "ft_transformer",
            ], start=1)],
            "requested_model": "xgboost",
        },
        "execution": {"status": "success", "selected_model": selected_model, "selected_rank": selected_rank, "fallback_used": fallback, "attempts": attempts},
        "predictions": predictions,
        "evaluation": {"predictive_metrics": predictive_metrics, "calibration_metrics": {}, "probability_metadata": {}},
        "artifacts": {"model_artifact": None, "model_reference": "fixture.Model", "preprocessor_artifact": None, "preprocessor_reference": "fixture.Preprocessor", "target_encoder_artifact": None, "target_encoder_reference": None, "transformed_feature_names": ["feature"]},
        "warnings": [], "errors": [],
    }


class _Outcome:
    def __init__(self, payload):
        self.result = payload
        self.runtime = object()

    def to_dict(self):
        return self.result


class PredictionExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.settings = ApplicationSettings(
            runtime_root=Path(self.temporary.name) / "runtime",
            maximum_upload_bytes=1024 * 1024,
            pipeline1_worker_count=1,
            prediction_worker_count=2,
            run_retention_seconds=60,
        )
        self.run_service = ApplicationRunService(
            self.settings, pipeline1_runner=_write_successful_pipeline1_artifacts
        )
        self.services = []

    def tearDown(self):
        for service in self.services:
            service.shutdown(wait=True)
        self.run_service.shutdown(wait=True)
        self.temporary.cleanup()

    def completed_run(self, content=None):
        content = content or b"feature,target\n1,no\n2,yes\n3,no\n4,yes\n5,no\n6,yes\n7,no\n8,yes\n"
        from starlette.datastructures import Headers, UploadFile
        from io import BytesIO
        import asyncio

        upload = UploadFile(BytesIO(content), filename="data.csv", headers=Headers({"content-type": "text/csv"}))
        created = asyncio.run(self.run_service.create_from_upload(upload))
        run_id = created["run_id"]
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if self.run_service.store.read(run_id)["state"] == "complete":
                return run_id
            time.sleep(0.01)
        self.fail("Pipeline 1 fixture did not complete")

    def service(self, resolver=None, orchestrator=None, **kwargs):
        service = PredictionJobService(
            self.run_service,
            resolver=resolver or (lambda frame, target, report: _resolution()),
            orchestrator=orchestrator or (lambda frame, **options: _Outcome(_orchestration(options["task_type"]))),
            **kwargs,
        )
        self.services.append(service)
        return service

    @staticmethod
    def wait_final(service, run_id, prediction_id, timeout=4):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = service.status(run_id, prediction_id)
            if state["state"] in {"complete", "failed"}:
                return state
            time.sleep(0.01)
        raise AssertionError("prediction did not finish")

    def test_queued_transitions_to_running_task_resolution_and_receives_canonical_inputs(self):
        run_id = self.completed_run()
        entered = Event()
        release = Event()
        observed = {}

        def resolver(frame, target, report):
            observed.update(frame=frame.copy(), target=target, report=report)
            entered.set()
            release.wait(2)
            return _resolution()

        service = self.service(resolver=resolver)
        created = service.create(run_id, "target")
        self.assertEqual(created["state"], "queued")
        self.assertTrue(entered.wait(2))
        running = service.status(run_id, created["prediction_id"])
        self.assertEqual((running["state"], running["stage"]), ("running", "task_resolution"))
        self.assertEqual(observed["target"], "target")
        self.assertEqual(list(observed["frame"].columns), ["feature", "target"])
        canonical = json.loads((self.run_service.store.paths(run_id)["pipeline1"] / "cleaning_report.json").read_text(encoding="utf-8"))
        self.assertEqual(observed["report"], canonical)
        release.set()
        self.assertEqual(self.wait_final(service, run_id, created["prediction_id"])["state"], "complete")

    def test_resolved_classification_and_regression_proceed(self):
        for task_type in ("classification", "regression"):
            run_id = self.completed_run()
            calls = []
            service = self.service(
                resolver=lambda frame, target, report, task=task_type: _resolution(task),
                orchestrator=lambda frame, **options: calls.append(options) or _Outcome(_orchestration(options["task_type"])),
            )
            created = service.create(run_id, "target")
            final = self.wait_final(service, run_id, created["prediction_id"])
            self.assertEqual(final["state"], "complete", final)
            self.assertEqual(calls[0]["task_type"], task_type)
            self.assertFalse(calls[0]["generate_semantic_profile"])

    def test_unresolved_fails_without_orchestrator(self):
        run_id = self.completed_run()
        calls = []
        service = self.service(
            resolver=lambda frame, target, report: _resolution(status="unresolved"),
            orchestrator=lambda *args, **kwargs: calls.append(1),
        )
        created = service.create(run_id, "target")
        final = self.wait_final(service, run_id, created["prediction_id"])
        self.assertEqual(final["error"]["code"], "task_type_unresolved")
        self.assertEqual(final["task_resolution"]["status"], "unresolved")
        self.assertEqual(calls, [])

    def test_two_orchestrator_failures_are_controlled_at_real_failure_stage(self):
        run_id = self.completed_run()
        failed = _orchestration()
        failed["execution"] = {
            "status": "failed", "selected_model": None,
            "selected_rank": None, "fallback_used": False,
            "attempts": [
                {"rank": 1, "model_id": "xgboost", "status": "failed", "failure_stage": "model_fit", "error": "raw one"},
                {"rank": 2, "model_id": "lightgbm", "status": "failed", "failure_stage": "model_fit", "error": "raw two"},
            ],
        }
        failed["predictions"] = None
        failed["evaluation"] = None
        service = self.service(orchestrator=lambda frame, **options: _Outcome(failed))
        created = service.create(run_id, "target")
        final = self.wait_final(service, run_id, created["prediction_id"])
        self.assertEqual(final["stage"], "training")
        self.assertEqual(final["error"]["code"], "top2_fallback_technical_failure")
        self.assertNotIn("raw one", json.dumps(final))
        self.assertNotIn("raw two", json.dumps(final))

    def test_changed_hash_after_queue_fails_before_resolver(self):
        run_id = self.completed_run()
        calls = []
        service = self.service(resolver=lambda *args: calls.append(1), auto_execute=False)
        created = service.create(run_id, "target")
        cleaned = self.run_service.store.paths(run_id)["pipeline1"] / "cleaned_dataset.csv"
        cleaned.write_text("feature,target\n9,no\n", encoding="utf-8")
        service._execute(run_id, created["prediction_id"])
        final = service.status(run_id, created["prediction_id"])
        self.assertEqual(final["error"]["code"], "cleaned_artifact_hash_mismatch")
        self.assertEqual(calls, [])

    def test_target_disappeared_at_execution_fails_safely(self):
        run_id = self.completed_run()
        service = self.service(auto_execute=False)
        created = service.create(run_id, "target")
        paths = self.run_service.store.paths(run_id)
        cleaned = paths["pipeline1"] / "cleaned_dataset.csv"
        cleaned.write_text("feature,other\n1,no\n2,yes\n3,no\n4,yes\n", encoding="utf-8")
        report_path = paths["pipeline1"] / "cleaning_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["decision_summary"]["privacy_actions"]["cleaned_dataset_sha256"] = hashlib.sha256(cleaned.read_bytes()).hexdigest()
        report_path.write_text(json.dumps(report), encoding="utf-8")
        service._execute(run_id, created["prediction_id"])
        final = service.status(run_id, created["prediction_id"])
        self.assertEqual(final["error"]["code"], "target_disappeared")

    def test_selector_integrity_failure_is_controlled(self):
        run_id = self.completed_run()
        service = self.service(selector_path=Path(self.temporary.name) / "missing.joblib")
        created = service.create(run_id, "target")
        final = self.wait_final(service, run_id, created["prediction_id"])
        self.assertEqual((final["stage"], final["error"]["code"]), ("model_selection", "selector_integrity_failure"))

    def test_success_persists_safe_result_csv_and_no_live_handles(self):
        run_id = self.completed_run()
        service = self.service(orchestrator=lambda frame, **options: _Outcome(_orchestration(fallback=True)))
        created = service.create(run_id, "target")
        prediction_id = created["prediction_id"]
        final = self.wait_final(service, run_id, prediction_id)
        self.assertEqual((final["state"], final["stage"]), ("complete", "complete"))
        internal = service.store.read_result(run_id, prediction_id)
        self.assertEqual(internal["selection"]["selected_rank"], 2)
        self.assertTrue(internal["selection"]["fallback_used"])
        result = service.result(run_id, prediction_id)
        self.assertNotIn("selection", result)
        serialized = json.dumps(internal)
        self.assertNotIn("SECRET", serialized)
        self.assertNotIn("artifact_path", serialized)
        self.assertNotIn("artifact_sha256", serialized)
        self.assertNotIn("runtime handles", serialized)
        directory = self.run_service.store.paths(run_id)["run"] / "predictions" / prediction_id
        self.assertTrue((directory / "result.json").is_file())
        self.assertTrue((directory / "predictions.csv").is_file())
        self.assertEqual(list(directory.glob("*.tmp")), [])

    def test_complete_only_after_result_persistence_and_serialization_failures(self):
        class BrokenStore(PredictionStore):
            def write_result(self, *args, **kwargs):
                state = self.read(args[0], args[1])
                self.observed_state = state["state"]
                raise OSError("disk detail")

        run_id = self.completed_run()
        store = BrokenStore(self.run_service.store)
        service = PredictionJobService(
            self.run_service, store=store,
            resolver=lambda *args: _resolution(),
            orchestrator=lambda frame, **options: _Outcome(_orchestration()),
        )
        self.services.append(service)
        created = service.create(run_id, "target")
        final = self.wait_final(service, run_id, created["prediction_id"])
        self.assertEqual(store.observed_state, "running")
        self.assertEqual(final["error"]["code"], "result_serialization_failure")
        self.assertNotIn("disk detail", json.dumps(final))

        second_run = self.completed_run()
        bad = _orchestration()
        bad["evaluation"]["predictive_metrics"]["balanced_accuracy"] = float("nan")
        second = self.service(orchestrator=lambda frame, **options: _Outcome(bad))
        created = second.create(second_run, "target")
        final = self.wait_final(second, second_run, created["prediction_id"])
        self.assertEqual(final["error"]["code"], "result_serialization_failure")

    def test_http_status_result_and_concurrent_jobs_are_isolated(self):
        first_run, second_run = self.completed_run(), self.completed_run()
        lock = Lock()
        observed = []

        def orchestrator(frame, **options):
            with lock:
                observed.append(options["dataset_id"])
            return _Outcome(_orchestration(options["task_type"]))

        service = self.service(orchestrator=orchestrator, worker_count=2)
        client = TestClient(create_app(service=self.run_service, prediction_service=service))
        first = client.post(f"/api/runs/{first_run}/predictions", json={"target_column": "target"}).json()
        second = client.post(f"/api/runs/{second_run}/predictions", json={"target_column": "target"}).json()
        for run_id, created in ((first_run, first), (second_run, second)):
            final = self.wait_final(service, run_id, created["prediction_id"])
            self.assertEqual(final["state"], "complete")
            response = client.get(f"/api/runs/{run_id}/predictions/{created['prediction_id']}/result")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["prediction_id"], created["prediction_id"])
            self.assertEqual(response.json()["status"], "complete")
        self.assertEqual(set(observed), {first_run, second_run})


if __name__ == "__main__":
    unittest.main()
