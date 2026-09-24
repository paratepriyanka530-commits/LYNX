"""Stage 5A: deterministic, non-replaying prediction restart recovery."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Barrier, Event, Lock
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from api.prediction_service import INTERRUPTED_MESSAGE, PredictionJobService
from api.run_service import ApplicationRunService, ServiceError
from api.server import create_app
from app.config import ApplicationSettings
from tests.api.test_application_flow import _write_successful_pipeline1_artifacts
from tests.api.test_prediction_frontend_api import _internal_result


class PredictionRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "runtime"
        self.run_service = ApplicationRunService(
            ApplicationSettings(runtime_root=self.root, prediction_worker_count=1),
            pipeline1_runner=_write_successful_pipeline1_artifacts,
        )
        self.services = []

    def tearDown(self):
        for service in self.services:
            service.shutdown(wait=True)
        self.run_service.shutdown(wait=True)
        self.temporary.cleanup()

    def service(self, *, auto_execute=False):
        service = PredictionJobService(self.run_service, auto_execute=auto_execute)
        self.services.append(service)
        return service

    def completed_run(self):
        with TestClient(create_app(service=self.run_service, prediction_service=self.service())) as client:
            response = client.post(
                "/api/runs", files={"dataset": ("data.csv", b"feature,target\n1,yes\n2,no\n3,yes\n", "text/csv")},
            )
            self.assertEqual(response.status_code, 202)
            run_id = response.json()["run_id"]
            from time import monotonic, sleep
            deadline = monotonic() + 3
            while monotonic() < deadline:
                status = client.get(f"/api/runs/{run_id}").json()["state"]
                if status == "complete":
                    return run_id
                self.assertNotEqual(status, "failed")
                sleep(0.01)
        self.fail("Fixture run did not complete")

    def test_queued_restart_is_failed_once_without_resubmission(self):
        run_id = self.completed_run()
        old = self.service()
        job = old.create(run_id, "target")
        replacement = self.service(auto_execute=True)
        with patch.object(replacement, "_schedule", side_effect=AssertionError("replayed")):
            first = replacement.reconcile_startup()
            second = replacement.reconcile_startup()
        state = replacement.status(run_id, job["prediction_id"])
        self.assertEqual(first["queued_interrupted"], 1)
        self.assertEqual(second["queued_interrupted"], 0)
        self.assertEqual(state["state"], "failed")
        self.assertEqual(state["error"], {"code": "prediction_interrupted", "message": INTERRUPTED_MESSAGE})
        self.assertEqual(replacement._active_predictions, set())
        self.assertNotIn("worker_instance_id", state)

    def test_current_instance_queued_job_is_not_mistaken_for_orphan(self):
        run_id = self.completed_run()
        service = self.service()
        job = service.create(run_id, "target")
        self.assertEqual(service.reconcile_startup()["queued_interrupted"], 0)
        self.assertEqual(service.status(run_id, job["prediction_id"])["state"], "queued")

    def test_old_running_becomes_failed_with_safe_error_and_preserved_start(self):
        run_id = self.completed_run()
        old = self.service()
        job = old.create(run_id, "target")
        old.store.transition_if(
            run_id, job["prediction_id"], expected={"queued"},
            state="running", stage="training", worker_instance_id=old.instance_id,
            started_at="2026-01-01T00:00:00+00:00",
        )
        replacement = self.service()
        self.assertEqual(replacement.reconcile_startup()["running_interrupted"], 1)
        state = replacement.status(run_id, job["prediction_id"])
        self.assertEqual(state["state"], "failed")
        self.assertEqual(state["error"]["code"], "prediction_interrupted")
        self.assertEqual(state["started_at"], "2026-01-01T00:00:00+00:00")
        self.assertIn("interrupted_at", state)
        self.assertNotIn(str(self.root), json.dumps(state))

    def test_final_states_unchanged_and_missing_complete_artifact_is_read_failure(self):
        run_id = self.completed_run()
        old = self.service()
        completed = old.create(run_id, "target")["prediction_id"]
        failed = old.create(run_id, "target")["prediction_id"]
        old.store.write_result(run_id, completed, _internal_result(completed))
        old.store.write_prediction_rows(run_id, completed, ["actual", "prediction"], [["yes", "yes"]])
        old.store.update(run_id, completed, state="complete", stage="complete")
        old.store.update(run_id, failed, state="failed", stage="training", error={"code": "controlled", "message": "Failed."})
        before = {pid: old.store._paths(run_id, pid)["state"].read_bytes() for pid in (completed, failed)}
        result_before = old.store._paths(run_id, completed)["result"].read_bytes()
        replacement = self.service()
        replacement.reconcile_startup()
        for pid in (completed, failed):
            self.assertEqual(old.store._paths(run_id, pid)["state"].read_bytes(), before[pid])
        self.assertEqual(old.store._paths(run_id, completed)["result"].read_bytes(), result_before)
        self.assertEqual(replacement.result(run_id, completed)["status"], "complete")
        self.assertEqual(replacement.rows(run_id, completed)["total"], 1)
        old.store._paths(run_id, completed)["result"].unlink()
        self.assertEqual(replacement.reconcile_startup()["invalid_entries"], 1)
        self.assertEqual(old.store._paths(run_id, completed)["state"].read_bytes(), before[completed])
        with self.assertRaises(ServiceError) as context:
            replacement.result(run_id, completed)
        self.assertEqual(context.exception.code, "prediction_result_unavailable")
        old.store._paths(run_id, completed)["result"].write_bytes(result_before)
        old.store._paths(run_id, completed)["predictions"].unlink()
        with self.assertRaises(ServiceError) as context:
            replacement.result(run_id, completed)
        self.assertEqual(context.exception.code, "prediction_rows_unavailable")
        self.assertEqual(old.store._paths(run_id, completed)["state"].read_bytes(), before[completed])

    def test_multiple_jobs_and_runs_recover_independently(self):
        first, second = self.completed_run(), self.completed_run()
        old = self.service()
        jobs = [(first, old.create(first, "target")["prediction_id"]),
                (first, old.create(first, "target")["prediction_id"]),
                (second, old.create(second, "target")["prediction_id"])]
        replacement = self.service()
        self.assertEqual(replacement.reconcile_startup()["queued_interrupted"], 3)
        self.assertEqual([replacement.status(run_id, pid)["state"] for run_id, pid in jobs], ["failed"] * 3)

    def test_concurrent_reconciliation_is_idempotent(self):
        run_id = self.completed_run()
        pid = self.service().create(run_id, "target")["prediction_id"]
        replacement = self.service()
        barrier = Barrier(3)
        def reconcile():
            barrier.wait()
            return replacement.reconcile_startup()
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(reconcile) for _ in range(2)]
            barrier.wait()
            changed = sum(future.result()["queued_interrupted"] for future in futures)
        self.assertEqual(changed, 1)
        self.assertEqual(replacement.status(run_id, pid)["state"], "failed")

    def test_corrupt_unknown_and_missing_entries_do_not_abort_startup(self):
        run_id = self.completed_run()
        old = self.service()
        valid = old.create(run_id, "target")["prediction_id"]
        root = self.root / run_id / "predictions"
        (root / "not-a-uuid").mkdir()
        (root / "not-a-uuid" / "state.json").write_text("{}", encoding="utf-8")
        malformed = root / str(uuid4())
        malformed.mkdir()
        (malformed / "state.json").write_text("{", encoding="utf-8")
        (root / str(uuid4())).mkdir()
        (self.root / "unknown-entry").mkdir()
        replacement = self.service()
        summary = replacement.reconcile_startup()
        self.assertEqual(summary["queued_interrupted"], 1)
        self.assertGreaterEqual(summary["invalid_entries"], 2)
        self.assertEqual(replacement.status(run_id, valid)["state"], "failed")
        self.assertTrue(malformed.exists())

    def test_lifespan_runs_recovery_and_failed_result_api_remains_controlled(self):
        run_id = self.completed_run()
        pid = self.service().create(run_id, "target")["prediction_id"]
        replacement = self.service()
        with TestClient(create_app(service=self.run_service, prediction_service=replacement)) as client:
            status = client.get(f"/api/runs/{run_id}/predictions/{pid}")
            result = client.get(f"/api/runs/{run_id}/predictions/{pid}/result")
            self.assertEqual(status.json()["state"], "failed")
            self.assertEqual(status.json()["error"]["code"], "prediction_interrupted")
            self.assertEqual(result.status_code, 409)
            self.assertEqual(result.json()["error"]["code"], "prediction_failed")

    def test_scan_never_imports_resolver_or_models(self):
        run_id = self.completed_run()
        self.service().create(run_id, "target")
        replacement = self.service()
        with patch("api.prediction_service.importlib.import_module", side_effect=AssertionError("ML import")):
            self.assertEqual(replacement.reconcile_startup()["queued_interrupted"], 1)

    def test_recovery_writes_only_beneath_runtime_root(self):
        run_id = self.completed_run()
        self.service().create(run_id, "target")
        replacement = self.service()
        written = []
        original = replacement.store._atomic_json
        def record_write(path, payload):
            written.append(path.resolve())
            return original(path, payload)
        with patch.object(replacement.store, "_atomic_json", side_effect=record_write):
            replacement.reconcile_startup()
        self.assertEqual(len(written), 1)
        for path in written:
            self.assertTrue(path.is_relative_to(self.root.resolve()))

    def test_normal_worker_persists_instance_ownership_and_rechecks_eligibility(self):
        run_id = self.completed_run()
        entered, release = Event(), Event()
        def resolver(_frame, _target, _report):
            entered.set()
            release.wait(timeout=3)
            return {"status": "unresolved", "task_type": None}
        service = PredictionJobService(self.run_service, resolver=resolver, auto_execute=True)
        self.services.append(service)
        created = service.create(run_id, "target")
        self.assertTrue(entered.wait(timeout=3))
        running = service.status(run_id, created["prediction_id"])
        self.assertEqual(running["state"], "running")
        self.assertEqual(running["worker_instance_id"], service.instance_id)
        self.assertIn("started_at", running)
        release.set()
        service.shutdown(wait=True)
        self.assertEqual(service.status(run_id, created["prediction_id"])["state"], "failed")

        # The same execution path checks integrity again rather than trusting
        # the eligibility checked when the prediction was originally created.
        second = self.service()
        queued = second.create(run_id, "target")
        cleaned = self.run_service.store.paths(run_id)["pipeline1"] / "cleaned_dataset.csv"
        cleaned.write_text("feature,target\n99,tampered\n", encoding="utf-8")
        second._execute(run_id, queued["prediction_id"])
        failed = second.status(run_id, queued["prediction_id"])
        self.assertEqual(failed["error"]["code"], "cleaned_artifact_hash_mismatch")

    def test_same_id_cannot_be_scheduled_twice_in_one_instance(self):
        run_id = self.completed_run()
        service = self.service(auto_execute=True)
        pid = service.store.create(run_id, "target")["prediction_id"]
        entered, release = Event(), Event()
        observed = []
        guard = Lock()
        def blocked(_run_id, _prediction_id):
            with guard:
                observed.append((_run_id, _prediction_id))
            entered.set()
            release.wait(timeout=3)
        with patch.object(service, "_execute", side_effect=blocked):
            self.assertTrue(service._schedule(run_id, pid))
            self.assertTrue(entered.wait(timeout=3))
            self.assertFalse(service._schedule(run_id, pid))
            release.set()
            service.shutdown(wait=True)
        self.assertEqual(observed, [(run_id, pid)])
        self.assertEqual(service._active_predictions, set())

    def test_shutdown_returns_without_waiting_and_restart_reconciles_pending_work(self):
        run_id = self.completed_run()
        service = self.service(auto_execute=True)
        first = service.store.create(run_id, "target")["prediction_id"]
        second = service.store.create(run_id, "target")["prediction_id"]
        entered, release, shutdown_returned = Event(), Event(), Event()
        def blocked(_run_id, _prediction_id):
            entered.set()
            release.wait(timeout=3)
        with patch.object(service, "_execute", side_effect=blocked):
            self.assertTrue(service._schedule(run_id, first))
            self.assertTrue(entered.wait(timeout=3))
            self.assertTrue(service._schedule(run_id, second))
            with ThreadPoolExecutor(max_workers=1) as caller:
                future = caller.submit(lambda: (service.shutdown(wait=False), shutdown_returned.set()))
                self.assertTrue(shutdown_returned.wait(timeout=3))
                release.set()
                future.result()
        replacement = self.service()
        self.assertEqual(replacement.reconcile_startup()["queued_interrupted"], 2)


if __name__ == "__main__":
    unittest.main()
