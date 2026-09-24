import ast
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from threading import Barrier, Lock
import tempfile
import time
import unittest
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from api.prediction_service import PredictionJobService, PredictionStore
from api.run_service import ApplicationRunService, ServiceError
from api.server import create_app
from app.config import ApplicationSettings
from tests.api.test_application_flow import _write_successful_pipeline1_artifacts


class PredictionJobApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "runtime"
        self.settings = ApplicationSettings(
            runtime_root=self.root,
            maximum_upload_bytes=1024 * 1024,
            pipeline1_worker_count=1,
            run_retention_seconds=60,
            upload_chunk_bytes=7,
        )
        self.service = ApplicationRunService(
            self.settings,
            pipeline1_runner=_write_successful_pipeline1_artifacts,
        )
        self.prediction_service = PredictionJobService(
            self.service, auto_execute=False
        )
        self.client = TestClient(create_app(
            service=self.service, prediction_service=self.prediction_service
        ))

    def tearDown(self):
        self.service.shutdown(wait=True)
        self.prediction_service.shutdown(wait=True)
        self.temporary.cleanup()

    def completed_run(
        self,
        *,
        content=b"feature,target\n1,yes\n2,no\n3,yes\n",
        client=None,
    ):
        client = client or self.client
        response = client.post(
            "/api/runs",
            files={"dataset": ("data.csv", content, "text/csv")},
        )
        self.assertEqual(response.status_code, 202, response.text)
        run_id = response.json()["run_id"]
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            state = client.get(f"/api/runs/{run_id}").json()["state"]
            if state == "complete":
                return run_id
            self.assertNotEqual(state, "failed")
            time.sleep(0.01)
        self.fail("Pipeline 1 fixture did not complete")

    def create_prediction(self, run_id, target="target", client=None):
        client = client or self.client
        return client.post(
            f"/api/runs/{run_id}/predictions",
            json={"target_column": target},
        )

    def test_create_persists_queued_job_and_real_contract_round_trip(self):
        run_id = self.completed_run(
            content=b"feature,target\n1,SECRET_VALUE_A\n2,SECRET_VALUE_B\n3,SECRET_VALUE_A\n"
        )
        response = self.create_prediction(run_id)
        self.assertEqual(response.status_code, 202, response.text)
        created = response.json()
        prediction_id = created["prediction_id"]
        self.assertEqual(str(UUID(prediction_id)), prediction_id)
        self.assertEqual(created["state"], "queued")
        self.assertEqual(
            created["status_url"],
            f"/api/runs/{run_id}/predictions/{prediction_id}",
        )

        directory = self.service.store.paths(run_id)["run"] / "predictions" / prediction_id
        state_path = directory / "state.json"
        self.assertTrue(directory.is_dir())
        self.assertTrue(state_path.is_file())
        self.assertFalse((directory / "result.json").exists())
        persisted = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertEqual(persisted["target_column"], "target")
        self.assertEqual(persisted["state"], "queued")
        self.assertEqual(persisted["stage"], "queued")
        self.assertIsNone(persisted["error"])
        serialized = state_path.read_text(encoding="utf-8")
        self.assertNotIn("SECRET_VALUE_A", serialized)
        self.assertNotIn("SECRET_VALUE_B", serialized)

        status = self.client.get(created["status_url"])
        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.json(), persisted)
        result = self.client.get(f"{created['status_url']}/result")
        self.assertEqual(result.status_code, 409)
        self.assertEqual(result.json()["error"]["code"], "prediction_not_complete")

    def test_malformed_and_unknown_run_ids_are_controlled(self):
        malformed = self.create_prediction("not-a-uuid")
        unknown = self.create_prediction(str(uuid4()))
        self.assertEqual(malformed.status_code, 400)
        self.assertEqual(malformed.json()["error"]["code"], "malformed_run_id")
        self.assertEqual(unknown.status_code, 404)
        self.assertEqual(unknown.json()["error"]["code"], "run_not_found")

    def test_pipeline1_must_be_complete(self):
        state = self.service.store.create("data.csv", ".csv")
        response = self.create_prediction(state["application_run_id"])
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["code"], "pipeline1_not_complete")

    def test_false_eligibility_is_blocked(self):
        run_id = self.completed_run()
        report_path = self.service.store.paths(run_id)["pipeline1"] / "cleaning_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["validation"]["passed"] = False
        report_path.write_text(json.dumps(report), encoding="utf-8")
        response = self.create_prediction(run_id)
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["code"], "prediction_not_eligible")

    def test_privacy_clearance_failure_is_blocked(self):
        run_id = self.completed_run()
        report_path = self.service.store.paths(run_id)["pipeline1"] / "cleaning_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["decision_summary"]["privacy_actions"]["privacy_clearance_passed"] = False
        report_path.write_text(json.dumps(report), encoding="utf-8")
        response = self.create_prediction(run_id)
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["code"], "prediction_not_eligible")

    def test_cleaned_dataset_hash_mismatch_is_blocked(self):
        run_id = self.completed_run()
        cleaned = self.service.store.paths(run_id)["pipeline1"] / "cleaned_dataset.csv"
        cleaned.write_text("feature,target\n999,tampered\n", encoding="utf-8")
        response = self.create_prediction(run_id)
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["code"], "prediction_not_eligible")

    def test_missing_cleaned_dataset_and_missing_expected_hash_are_blocked(self):
        missing_dataset_run = self.completed_run()
        cleaned = self.service.store.paths(missing_dataset_run)["pipeline1"] / "cleaned_dataset.csv"
        cleaned.unlink()
        missing_dataset = self.create_prediction(missing_dataset_run)
        self.assertEqual(missing_dataset.status_code, 409)
        self.assertEqual(missing_dataset.json()["error"]["code"], "prediction_not_eligible")

        missing_hash_run = self.completed_run()
        report_path = self.service.store.paths(missing_hash_run)["pipeline1"] / "cleaning_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["decision_summary"]["privacy_actions"].pop("cleaned_dataset_sha256")
        report_path.write_text(json.dumps(report), encoding="utf-8")
        missing_hash = self.create_prediction(missing_hash_run)
        self.assertEqual(missing_hash.status_code, 409)
        self.assertEqual(missing_hash.json()["error"]["code"], "prediction_not_eligible")

    def test_missing_empty_not_found_and_case_mismatch_targets_are_rejected(self):
        run_id = self.completed_run()
        missing = self.client.post(f"/api/runs/{run_id}/predictions", json={})
        empty = self.create_prediction(run_id, "   ")
        absent = self.create_prediction(run_id, "absent")
        case_mismatch = self.create_prediction(run_id, "Target")
        surrounding_space = self.create_prediction(run_id, " target ")
        self.assertEqual((missing.status_code, missing.json()["error"]["code"]), (422, "invalid_target"))
        self.assertEqual((empty.status_code, empty.json()["error"]["code"]), (422, "invalid_target"))
        self.assertEqual((absent.status_code, absent.json()["error"]["code"]), (422, "target_not_found"))
        self.assertEqual((case_mismatch.status_code, case_mismatch.json()["error"]["code"]), (422, "target_not_found"))
        self.assertEqual((surrounding_space.status_code, surrounding_space.json()["error"]["code"]), (422, "target_not_found"))

    def test_duplicate_prepared_headers_block_creation(self):
        service = ApplicationRunService(
            self.settings,
            pipeline1_runner=lambda source, output: _write_successful_pipeline1_artifacts(
                source, output, duplicate_columns=True
            ),
        )
        prediction_service = PredictionJobService(service, auto_execute=False)
        client = TestClient(create_app(
            service=service, prediction_service=prediction_service
        ))
        try:
            run_id = self.completed_run(client=client)
            response = self.create_prediction(run_id, "target", client=client)
            self.assertEqual(response.status_code, 409)
            self.assertEqual(response.json()["error"]["code"], "prediction_not_eligible")
        finally:
            prediction_service.shutdown(wait=True)
            service.shutdown(wait=True)

    def test_path_like_targets_never_influence_filesystem_paths(self):
        run_id = self.completed_run()
        run_directory = self.service.store.paths(run_id)["run"]
        for target in ("../../something", r"C:\foo", "a/b", r"..\target"):
            response = self.create_prediction(run_id, target)
            self.assertEqual(response.status_code, 422)
            self.assertEqual(response.json()["error"]["code"], "target_not_found")
        prediction_root = run_directory / "predictions"
        self.assertFalse(prediction_root.exists())
        self.assertFalse((self.root / "something").exists())

        path_header_run = self.completed_run(
            content=b"../../something,feature\nsecret-a,1\nsecret-b,2\nsecret-c,3\n"
        )
        accepted = self.create_prediction(path_header_run, "../../something")
        self.assertEqual(accepted.status_code, 202, accepted.text)
        prediction_id = accepted.json()["prediction_id"]
        self.assertEqual(str(UUID(prediction_id)), prediction_id)
        accepted_root = self.service.store.paths(path_header_run)["run"] / "predictions"
        self.assertEqual([path.name for path in accepted_root.iterdir()], [prediction_id])
        self.assertFalse((self.root / "something").exists())

    def test_prediction_lookup_validates_ids_and_run_ownership(self):
        first_run = self.completed_run()
        second_run = self.completed_run()
        prediction_id = self.create_prediction(first_run).json()["prediction_id"]
        malformed = self.client.get(f"/api/runs/{first_run}/predictions/not-a-uuid")
        unknown = self.client.get(f"/api/runs/{first_run}/predictions/{uuid4()}")
        wrong_run = self.client.get(f"/api/runs/{second_run}/predictions/{prediction_id}")
        malformed_run = self.client.get(f"/api/runs/not-a-uuid/predictions/{prediction_id}")
        self.assertEqual((malformed.status_code, malformed.json()["error"]["code"]), (400, "malformed_prediction_id"))
        self.assertEqual((unknown.status_code, unknown.json()["error"]["code"]), (404, "prediction_not_found"))
        self.assertEqual((wrong_run.status_code, wrong_run.json()["error"]["code"]), (404, "prediction_not_found"))
        self.assertEqual((malformed_run.status_code, malformed_run.json()["error"]["code"]), (400, "malformed_run_id"))

    def test_duplicate_requests_create_independent_prediction_ids(self):
        run_id = self.completed_run()
        first = self.create_prediction(run_id).json()["prediction_id"]
        second = self.create_prediction(run_id).json()["prediction_id"]
        self.assertNotEqual(first, second)
        root = self.service.store.paths(run_id)["run"] / "predictions"
        self.assertEqual({path.name for path in root.iterdir()}, {first, second})

    def test_complete_and_failed_result_contracts_are_reserved(self):
        from tests.api.test_prediction_frontend_api import _internal_result

        run_id = self.completed_run()
        prediction_id = self.create_prediction(run_id).json()["prediction_id"]
        store = self.client.app.state.prediction_service.store
        store.write_result(run_id, prediction_id, _internal_result(prediction_id))
        store.write_prediction_rows(run_id, prediction_id, ["actual", "prediction"], [["yes", "yes"]])
        store.update(run_id, prediction_id, state="complete", stage="complete")
        complete = self.client.get(f"/api/runs/{run_id}/predictions/{prediction_id}/result")
        self.assertEqual(complete.status_code, 200)
        self.assertEqual(complete.json()["status"], "complete")
        self.assertNotIn("selection", complete.json())

        failed_id = self.create_prediction(run_id).json()["prediction_id"]
        store.update(
            run_id,
            failed_id,
            state="failed",
            stage="training",
            error={"code": "future_failure", "message": "Controlled failure."},
        )
        failed = self.client.get(f"/api/runs/{run_id}/predictions/{failed_id}/result")
        self.assertEqual(failed.status_code, 409)
        self.assertEqual(failed.json()["error"]["code"], "prediction_failed")

    def test_run_cleanup_removes_nested_predictions(self):
        run_id = self.completed_run()
        prediction_id = self.create_prediction(run_id).json()["prediction_id"]
        prediction_directory = self.service.store.paths(run_id)["run"] / "predictions" / prediction_id
        self.assertTrue(prediction_directory.is_dir())
        future = datetime.now(timezone.utc) + timedelta(days=2)
        removed = self.service.store.cleanup_expired(1, now=future)
        self.assertEqual(removed, [run_id])
        self.assertFalse(prediction_directory.exists())


class PredictionStoreConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.run_store_service = ApplicationRunService(
            ApplicationSettings(runtime_root=Path(self.temporary.name) / "runtime"),
            pipeline1_runner=_write_successful_pipeline1_artifacts,
        )
        self.store = PredictionStore(self.run_store_service.store)

    def tearDown(self):
        self.run_store_service.shutdown(wait=True)
        self.temporary.cleanup()

    def new_run(self):
        state = self.run_store_service.store.create("data.csv", ".csv")
        return state["application_run_id"]

    def test_concurrent_reads_and_updates_never_expose_partial_json(self):
        run_id = self.new_run()
        prediction_id = self.store.create(run_id, "target")["prediction_id"]
        barrier = Barrier(5)
        failures = []
        observed = []
        result_lock = Lock()

        def writer():
            barrier.wait()
            for index in range(80):
                self.store.update(run_id, prediction_id, stage=f"future-stage-{index}")

        def reader():
            barrier.wait()
            for _index in range(200):
                try:
                    payload = self.store.read(run_id, prediction_id)
                    with result_lock:
                        observed.append(payload)
                except Exception as exc:  # pragma: no cover - assertion captures any corruption
                    with result_lock:
                        failures.append(type(exc).__name__)

        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(writer)] + [pool.submit(reader) for _ in range(4)]
            for future in futures:
                future.result()

        self.assertEqual(failures, [])
        self.assertTrue(observed)
        self.assertTrue(all(item["prediction_id"] == prediction_id for item in observed))
        self.assertTrue(all(item["run_id"] == run_id for item in observed))

    def test_predictions_for_different_runs_remain_isolated(self):
        first_run, second_run = self.new_run(), self.new_run()
        first = self.store.create(first_run, "target")
        second = self.store.create(second_run, "target")
        self.assertNotEqual(first["prediction_id"], second["prediction_id"])
        self.assertEqual(self.store.read(first_run, first["prediction_id"])["run_id"], first_run)
        self.assertEqual(self.store.read(second_run, second["prediction_id"])["run_id"], second_run)
        with self.assertRaises(ServiceError) as context:
            self.store.read(second_run, first["prediction_id"])
        self.assertEqual(context.exception.code, "prediction_not_found")


class LightweightImportTests(unittest.TestCase):
    def test_stage_4a_modules_do_not_import_future_prediction_components(self):
        forbidden = {
            "torch",
            "tabpfn",
            "tabicl",
            "joblib",
            "api.task_type_resolver",
            "pipelines.pipeline2_prediction.orchestration.prediction_orchestrator",
        }
        imported = set()
        for relative in ("api/server.py", "api/prediction_service.py"):
            tree = ast.parse(Path(relative).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module)
        self.assertTrue(forbidden.isdisjoint(imported), forbidden.intersection(imported))


if __name__ == "__main__":
    unittest.main()
