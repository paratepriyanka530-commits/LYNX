import hashlib
from io import BytesIO
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
from threading import Barrier, Event, Lock
import time
import unittest
from uuid import UUID
import sys

import pandas as pd
from openpyxl import Workbook
from fastapi.testclient import TestClient

from api.run_service import (
    ApplicationRunService,
    RunStore,
    ServiceError,
    prediction_eligibility,
)
from api.server import create_app
from app.config import ApplicationSettings
from pipelines.pipeline1_cleaning.ingestion.dataset_loader import load_dataset
from pipelines.pipeline1_cleaning.ingestion.file_validator import validate_file


class SpreadsheetParserTests(unittest.TestCase):
    def test_real_sales_data_xlsx_validation_and_loader_path(self):
        path = Path(__file__).resolve().parents[1] / "datasets" / "valid" / "Sales_Data.xlsx"
        validation = validate_file(str(path))
        self.assertTrue(validation["valid"], validation)
        self.assertEqual((validation["rows"], validation["columns"]), (100, 8))
        loaded = load_dataset(str(path))
        self.assertEqual((loaded["rows"], loaded["columns"]), (100, 8))

    def test_xlsx_validation_and_loader_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "Sales_Data.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.append(["feature", "target"])
            sheet.append([1, "yes"])
            sheet.append([2, "no"])
            workbook.save(path)
            self.assertTrue(validate_file(str(path))["valid"])
            loaded = load_dataset(str(path))
            self.assertEqual((loaded["rows"], loaded["columns"]), (2, 2))

    def test_xls_validation_and_loader_path(self):
        fixture = Path(__file__).resolve().parents[1] / "datasets" / "valid" / "minimal_legacy.xls"
        self.assertTrue(validate_file(str(fixture))["valid"])
        loaded = load_dataset(str(fixture))
        self.assertEqual((loaded["rows"], loaded["columns"]), (2, 2))


class RunStoreConcurrencyTests(unittest.TestCase):
    def test_same_run_concurrent_reads_and_updates_are_consistent(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = RunStore(Path(temporary) / "runtime")
            created = store.create("source.csv", ".csv")
            run_id = created["application_run_id"]
            start = Barrier(5)
            failures = []
            observed = []
            result_lock = Lock()

            def writer():
                start.wait()
                for index in range(80):
                    store.update(
                        run_id,
                        state="running",
                        stage="pipeline1",
                        message=f"transition-{index}",
                    )
                store.update(
                    run_id,
                    state="complete",
                    stage="pipeline1_complete",
                    message="complete",
                    pipeline1_internal_run_id="internal-final",
                )

            def reader():
                start.wait()
                for _index in range(300):
                    try:
                        payload = store.read(run_id)
                        with result_lock:
                            observed.append(payload)
                    except ServiceError as exc:
                        with result_lock:
                            failures.append(exc.code)

            with ThreadPoolExecutor(max_workers=5) as pool:
                futures = [pool.submit(writer)] + [pool.submit(reader) for _ in range(4)]
                for future in futures:
                    future.result()

            self.assertEqual(failures, [])
            self.assertTrue(observed)
            for payload in observed:
                self.assertEqual(payload["application_run_id"], run_id)
                self.assertEqual(payload["original_filename"], "source.csv")
                self.assertEqual(payload["stored_extension"], ".csv")
                self.assertIn(payload["state"], {"queued", "running", "complete"})
                self.assertIn("created_at", payload)
                self.assertIn("updated_at", payload)

            final = store.read(run_id)
            self.assertEqual(final["state"], "complete")
            self.assertEqual(final["pipeline1_internal_run_id"], "internal-final")
            self.assertEqual(final["original_filename"], "source.csv")

    def test_concurrent_runs_keep_state_isolated(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = RunStore(Path(temporary) / "runtime")
            runs = [store.create(f"source-{index}.csv", ".csv") for index in range(4)]
            start = Barrier(8)
            failures = []
            result_lock = Lock()

            def writer(index, run_id):
                start.wait()
                for iteration in range(30):
                    store.update(
                        run_id,
                        state="running",
                        stage="pipeline1",
                        message=f"run-{index}-transition-{iteration}",
                    )
                store.update(run_id, state="complete", message=f"run-{index}-complete")

            def reader(index, run_id):
                start.wait()
                for _iteration in range(120):
                    try:
                        payload = store.read(run_id)
                        if payload["application_run_id"] != run_id:
                            with result_lock:
                                failures.append("cross_run_state")
                        if payload["original_filename"] != f"source-{index}.csv":
                            with result_lock:
                                failures.append("metadata_changed")
                    except ServiceError as exc:
                        with result_lock:
                            failures.append(exc.code)

            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = []
                for index, state in enumerate(runs):
                    run_id = state["application_run_id"]
                    futures.extend((pool.submit(writer, index, run_id), pool.submit(reader, index, run_id)))
                for future in futures:
                    future.result()

            self.assertEqual(failures, [])
            for index, state in enumerate(runs):
                final = store.read(state["application_run_id"])
                self.assertEqual(final["state"], "complete")
                self.assertEqual(final["message"], f"run-{index}-complete")
                self.assertEqual(final["original_filename"], f"source-{index}.csv")

    def test_cleanup_cannot_remove_a_run_during_active_transition(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = RunStore(Path(temporary) / "runtime")
            future = datetime.now(timezone.utc) + timedelta(days=2)
            for index in range(20):
                state = store.create(f"active-{index}.csv", ".csv")
                run_id = state["application_run_id"]
                start = Barrier(2)

                def mark_running():
                    start.wait()
                    store.update(run_id, state="running", stage="pipeline1")

                def clean():
                    start.wait()
                    return store.cleanup_expired(1, now=future)

                with ThreadPoolExecutor(max_workers=2) as pool:
                    update_future = pool.submit(mark_running)
                    cleanup_future = pool.submit(clean)
                    update_future.result()
                    removed = cleanup_future.result()

                self.assertNotIn(run_id, removed)
                self.assertEqual(store.read(run_id)["state"], "running")


def _write_successful_pipeline1_artifacts(
    input_path, output_directory, *, duplicate_columns=False, privacy_clear=True,
    pii_columns=(),
):
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    cleaned = output / "cleaned_dataset.csv"
    if duplicate_columns:
        cleaned.write_text("feature,target,target\n1,yes,no\n2,no,yes\n", encoding="utf-8")
    else:
        pd.read_csv(input_path).to_csv(cleaned, index=False)
    digest = hashlib.sha256(cleaned.read_bytes()).hexdigest()
    report = {
        "pipeline_version": "test",
        "run_id": "pipeline1-internal-test",
        "status": "success",
        "original_dataset": {
            "file_name": Path(input_path).name,
            "absolute_path": str(Path(input_path).resolve()),
            "source_path": str(Path(input_path).resolve()),
        },
        "before_profile": {"shape": {"rows": 3, "columns": 2}},
        "issues": [],
        "quality_summary": {"total_issues_detected": 0},
        "semantic_profile": {"dataset_context": {}, "columns": []},
        "decision_summary": {"privacy_actions": {
            "privacy_stage_completed": True,
            "privacy_clearance_passed": privacy_clear,
            "cleaned_dataset_sha256": digest,
            "pii_candidate_columns": list(pii_columns),
        }},
        "decisions": [],
        "instructions": [],
        "executions": [],
        "validation": {"passed": True, "after_summary": {}},
        "warnings": [],
        "output_artifacts": {
            "cleaned_dataset": str(cleaned.resolve()),
            "cleaning_report": str((output / "cleaning_report.json").resolve()),
            "cleaned_dataset_format": "csv",
        },
    }
    (output / "cleaning_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


class ApplicationFlowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "runtime"
        self.settings = ApplicationSettings(
            runtime_root=self.root,
            maximum_upload_bytes=1024 * 1024,
            pipeline1_worker_count=1,
            run_retention_seconds=60,
            upload_chunk_bytes=7,
            xlsx_max_entries=100,
            xlsx_max_uncompressed_bytes=1024 * 1024,
            xlsx_max_compression_ratio=100,
        )
        self.service = ApplicationRunService(
            self.settings, pipeline1_runner=_write_successful_pipeline1_artifacts
        )
        self.client = TestClient(create_app(service=self.service))

    def tearDown(self):
        self.service.shutdown(wait=True)
        self.temporary.cleanup()

    def upload(self, content=b"feature,target\n1,yes\n2,no\n3,yes\n", filename="data.csv"):
        return self.client.post(
            "/api/runs", files={"dataset": (filename, content, "application/octet-stream")}
        )

    def wait_for(self, run_id, state="complete", timeout=3.0):
        deadline = time.monotonic() + timeout
        payload = None
        while time.monotonic() < deadline:
            response = self.client.get(f"/api/runs/{run_id}")
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            if payload["state"] == state:
                return payload
            if payload["state"] == "failed" and state != "failed":
                self.fail(payload)
            time.sleep(0.01)
        self.fail(f"run did not reach {state}: {payload}")

    def test_valid_csv_upload_creates_uuid_and_completes(self):
        response = self.upload()
        self.assertEqual(response.status_code, 202)
        payload = response.json()
        self.assertEqual(str(UUID(payload["run_id"])), payload["run_id"])
        self.assertEqual(payload["state"], "queued")
        self.wait_for(payload["run_id"])

    def test_valid_xlsx_upload_is_accepted_by_real_parser_path(self):
        workbook = Workbook()
        sheet = workbook.active
        sheet.append(["feature", "target"])
        sheet.append([1, "yes"])
        sheet.append([2, "no"])
        content = BytesIO()
        workbook.save(content)
        response = self.upload(content.getvalue(), "Sales_Data.xlsx")
        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual(response.json()["state"], "queued")

    def test_real_sales_data_xlsx_is_accepted_at_api_boundary(self):
        fixture = Path(__file__).resolve().parents[1] / "datasets" / "valid" / "Sales_Data.xlsx"
        response = self.upload(fixture.read_bytes(), fixture.name)
        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual(response.json()["state"], "queued")

    def test_valid_legacy_xls_upload_is_accepted_by_real_parser_path(self):
        fixture = Path(__file__).resolve().parents[1] / "datasets" / "valid" / "minimal_legacy.xls"
        response = self.upload(fixture.read_bytes(), "Sales_Data.xls")
        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual(response.json()["state"], "queued")

    def test_server_does_not_import_database_scaffold(self):
        self.assertNotIn("database.connection", sys.modules)

    def test_same_origin_frontend_is_served_without_shadowing_api(self):
        frontend = self.client.get("/")
        self.assertEqual(frontend.status_code, 200)
        self.assertIn('id="view-upload"', frontend.text)
        self.assertIn('id="view-export"', frontend.text)
        self.assertIn('src="apiClient.js?v=20261018-settings"', frontend.text)
        self.assertEqual(self.client.get("/api/runs/not-a-uuid").status_code, 400)

    def test_unsupported_extension_is_rejected_without_run(self):
        response = self.upload(b"value", "data.txt")
        self.assertEqual(response.status_code, 415)
        self.assertEqual(response.json()["error"]["code"], "unsupported_file_type")
        self.assertEqual(list(self.root.iterdir()), [])

    def test_zero_byte_file_is_rejected_without_partial_run(self):
        response = self.upload(b"")
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "empty_upload")
        self.assertEqual(list(self.root.iterdir()), [])

    def test_oversized_upload_is_rejected_without_partial_run(self):
        settings = ApplicationSettings(
            runtime_root=Path(self.temporary.name) / "small-runtime",
            maximum_upload_bytes=8,
            upload_chunk_bytes=3,
        )
        service = ApplicationRunService(settings, pipeline1_runner=_write_successful_pipeline1_artifacts)
        client = TestClient(create_app(service=service))
        try:
            response = client.post(
                "/api/runs", files={"dataset": ("large.csv", b"a,b\n1,2\n3,4\n", "text/csv")}
            )
            self.assertEqual(response.status_code, 413)
            self.assertEqual(response.json()["error"]["code"], "upload_too_large")
            self.assertEqual(list(settings.runtime_root.iterdir()), [])
        finally:
            service.shutdown(wait=True)

    def test_malformed_csv_is_controlled(self):
        response = self.upload(b"column\n\xff\xfe\xff\n")
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "invalid_dataset")

    def test_malformed_xlsx_is_controlled(self):
        response = self.upload(b"not-a-zip-workbook", "broken.xlsx")
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "invalid_spreadsheet")

    def test_two_same_name_uploads_are_isolated(self):
        first, second = self.upload(), self.upload()
        first_id, second_id = first.json()["run_id"], second.json()["run_id"]
        self.assertNotEqual(first_id, second_id)
        self.wait_for(first_id)
        self.wait_for(second_id)
        self.assertNotEqual(
            self.service.store.paths(first_id)["run"],
            self.service.store.paths(second_id)["run"],
        )

    def test_path_traversal_filename_is_metadata_only(self):
        response = self.upload(filename="../../outside.csv")
        run_id = response.json()["run_id"]
        status = self.wait_for(run_id)
        self.assertEqual(status["original_filename"], "outside.csv")
        upload_path = self.service.store.paths(run_id)["upload"] / "dataset.csv"
        upload_path.resolve().relative_to(self.root.resolve())
        self.assertTrue(upload_path.is_file())

    def test_exactly_one_dataset_file_is_required(self):
        response = self.client.post(
            "/api/runs",
            files=[
                ("dataset", ("a.csv", b"a\n1\n", "text/csv")),
                ("dataset", ("b.csv", b"b\n2\n", "text/csv")),
            ],
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "invalid_dataset_upload_count")

    def test_status_endpoint_reports_truthful_running_state(self):
        started, release = Event(), Event()

        def blocking_runner(input_path, output_directory):
            started.set()
            release.wait(3)
            return _write_successful_pipeline1_artifacts(input_path, output_directory)

        service = ApplicationRunService(self.settings, pipeline1_runner=blocking_runner)
        client = TestClient(create_app(service=service))
        try:
            response = client.post(
                "/api/runs", files={"dataset": ("data.csv", b"a,b\n1,x\n2,y\n", "text/csv")}
            )
            run_id = response.json()["run_id"]
            self.assertTrue(started.wait(1))
            status = client.get(f"/api/runs/{run_id}").json()
            self.assertEqual(status["state"], "running")
            self.assertEqual(status["stage"], "pipeline1")
            release.set()
        finally:
            release.set()
            service.shutdown(wait=True)

    def test_unknown_and_malformed_run_ids_are_controlled(self):
        malformed = self.client.get("/api/runs/not-a-uuid")
        unknown = self.client.get(f"/api/runs/{UUID(int=0)}")
        self.assertEqual(malformed.status_code, 400)
        self.assertEqual(malformed.json()["error"]["code"], "malformed_run_id")
        self.assertEqual(unknown.status_code, 404)
        self.assertEqual(unknown.json()["error"]["code"], "run_not_found")

    def test_pipeline1_failure_becomes_controlled_failed_state(self):
        def failing_runner(_input_path, _output_directory):
            raise RuntimeError("G:/private/path/secret failure")

        service = ApplicationRunService(self.settings, pipeline1_runner=failing_runner)
        client = TestClient(create_app(service=service))
        try:
            response = client.post(
                "/api/runs", files={"dataset": ("data.csv", b"a,b\n1,x\n2,y\n", "text/csv")}
            )
            run_id = response.json()["run_id"]
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                payload = client.get(f"/api/runs/{run_id}").json()
                if payload["state"] == "failed":
                    break
                time.sleep(0.01)
            self.assertEqual(payload["error"]["code"], "pipeline1_execution_failed")
            self.assertNotIn("private", json.dumps(payload))
        finally:
            service.shutdown(wait=True)

    def test_pipeline1_result_before_completion_is_conflict(self):
        started, release = Event(), Event()

        def blocking_runner(input_path, output_directory):
            started.set()
            release.wait(3)
            return _write_successful_pipeline1_artifacts(input_path, output_directory)

        service = ApplicationRunService(self.settings, pipeline1_runner=blocking_runner)
        client = TestClient(create_app(service=service))
        try:
            response = client.post(
                "/api/runs", files={"dataset": ("data.csv", b"a,b\n1,x\n2,y\n", "text/csv")}
            )
            run_id = response.json()["run_id"]
            self.assertTrue(started.wait(1))
            result = client.get(f"/api/runs/{run_id}/pipeline1")
            self.assertEqual(result.status_code, 409)
            self.assertEqual(result.json()["error"]["code"], "pipeline1_not_complete")
        finally:
            release.set()
            service.shutdown(wait=True)

    def test_successful_result_has_final_columns_and_is_eligible(self):
        run_id = self.upload().json()["run_id"]
        self.wait_for(run_id)
        response = self.client.get(f"/api/runs/{run_id}/pipeline1")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["prepared_dataset"]["target_columns"], ["feature", "target"])
        self.assertEqual(payload["prepared_dataset"]["rows"], 3)
        self.assertTrue(payload["prediction_eligibility"]["eligible"])
        self.assertTrue(payload["prediction_eligibility"]["cleaned_dataset_sha256_verified"])
        overview = payload["dataset_overview"]
        original = pd.DataFrame({"feature": [1, 2, 3], "target": ["yes", "no", "yes"]})
        self.assertEqual(
            overview["memory_footprint_bytes"],
            int(original.memory_usage(index=True, deep=True).sum()),
        )
        self.assertEqual(overview["memory_footprint_stage"], "original_ingestion")
        self.assertEqual(overview["raw_data_preview"]["columns"], ["feature", "target"])
        self.assertEqual(
            overview["raw_data_preview"]["rows"],
            [[1, "yes"], [2, "no"], [3, "yes"]],
        )
        manifest = payload["profiling_manifest"]
        self.assertEqual(manifest["dataset_name"], "data.csv")
        self.assertEqual(manifest["profile_status"], "unavailable")
        self.assertEqual(manifest["total_columns"], 0)
        self.assertEqual(manifest["columns"], [])
        for existing_field in (
            "run_id", "state", "pipeline1_internal_run_id", "pipeline1_report",
            "prepared_dataset", "privacy_clearance_passed", "prediction_eligibility",
        ):
            self.assertIn(existing_field, payload)

    def test_result_preview_redacts_pii_and_is_withheld_without_privacy_clearance(self):
        content = b"contact_email,amount\nalice@example.com,10\nbob@example.com,20\n"
        for privacy_clear in (True, False):
            with self.subTest(privacy_clear=privacy_clear):
                service = ApplicationRunService(
                    self.settings,
                    pipeline1_runner=lambda source, output, clear=privacy_clear: _write_successful_pipeline1_artifacts(
                        source,
                        output,
                        privacy_clear=clear,
                        pii_columns=("contact_email",),
                    ),
                )
                client = TestClient(create_app(service=service))
                try:
                    response = client.post(
                        "/api/runs",
                        files={"dataset": ("contacts.csv", content, "text/csv")},
                    )
                    run_id = response.json()["run_id"]
                    deadline = time.monotonic() + 3
                    while client.get(f"/api/runs/{run_id}").json()["state"] != "complete":
                        self.assertLess(time.monotonic(), deadline)
                        time.sleep(0.01)
                    preview = client.get(f"/api/runs/{run_id}/pipeline1").json()["dataset_overview"]["raw_data_preview"]
                    if privacy_clear:
                        self.assertTrue(preview["available"])
                        self.assertEqual(preview["privacy_redacted_columns"], ["contact_email"])
                        self.assertEqual(preview["rows"][0], ["[REDACTED]", 10])
                    else:
                        self.assertFalse(preview["available"])
                        self.assertEqual(preview["unavailable_reason"], "privacy_clearance_not_passed")
                        self.assertEqual(preview["rows"], [])
                finally:
                    service.shutdown(wait=True)

    def test_local_paths_are_absent_from_api_but_canonical_report_is_unchanged(self):
        run_id = self.upload().json()["run_id"]
        self.wait_for(run_id)
        payload = self.client.get(f"/api/runs/{run_id}/pipeline1").json()
        serialized = json.dumps(payload)
        self.assertNotIn(str(self.root), serialized)
        self.assertNotIn("source_path", serialized)
        self.assertNotIn("absolute_path", serialized)
        canonical = json.loads(
            (self.service.store.paths(run_id)["pipeline1"] / "cleaning_report.json")
            .read_text(encoding="utf-8")
        )
        self.assertIn("source_path", canonical["original_dataset"])

    def test_pipeline1_output_directories_are_isolated(self):
        ids = [self.upload().json()["run_id"] for _ in range(2)]
        for run_id in ids:
            self.wait_for(run_id)
        directories = [self.service.store.paths(run_id)["pipeline1"] for run_id in ids]
        self.assertEqual(len(set(directories)), 2)
        self.assertTrue(all((path / "cleaning_report.json").is_file() for path in directories))

    def test_hash_mismatch_blocks_prediction_eligibility(self):
        run_id = self.upload().json()["run_id"]
        self.wait_for(run_id)
        cleaned = self.service.store.paths(run_id)["pipeline1"] / "cleaned_dataset.csv"
        cleaned.write_text("feature,target\n999,tampered\n", encoding="utf-8")
        payload = self.client.get(f"/api/runs/{run_id}/pipeline1").json()
        eligibility = payload["prediction_eligibility"]
        self.assertFalse(eligibility["eligible"])
        self.assertIn("cleaned_dataset_hash_mismatch", eligibility["reasons"])

    def test_duplicate_prepared_headers_block_targets_and_eligibility(self):
        service = ApplicationRunService(
            self.settings,
            pipeline1_runner=lambda source, output: _write_successful_pipeline1_artifacts(
                source, output, duplicate_columns=True
            ),
        )
        client = TestClient(create_app(service=service))
        try:
            response = client.post(
                "/api/runs", files={"dataset": ("data.csv", b"a,b\n1,x\n2,y\n", "text/csv")}
            )
            run_id = response.json()["run_id"]
            deadline = time.monotonic() + 3
            while client.get(f"/api/runs/{run_id}").json()["state"] != "complete":
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.01)
            payload = client.get(f"/api/runs/{run_id}/pipeline1").json()
            self.assertEqual(payload["prepared_dataset"]["target_columns"], [])
            self.assertIn(
                "prepared_dataset_has_duplicate_columns",
                payload["prediction_eligibility"]["reasons"],
            )
        finally:
            service.shutdown(wait=True)

    def test_privacy_or_validation_failure_blocks_eligibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cleaned = root / "cleaned_dataset.csv"
            cleaned.write_text("a\n1\n", encoding="utf-8")
            digest = hashlib.sha256(cleaned.read_bytes()).hexdigest()
            report = {
                "status": "success",
                "validation": {"passed": False},
                "decision_summary": {"privacy_actions": {
                    "privacy_stage_completed": True,
                    "privacy_clearance_passed": False,
                    "cleaned_dataset_sha256": digest,
                }},
            }
            result = prediction_eligibility(report, cleaned, columns_unique=True)
            self.assertFalse(result["eligible"])
            self.assertIn("post_clean_validation_not_passed", result["reasons"])
            self.assertIn("privacy_clearance_not_passed", result["reasons"])

    def test_cleanup_rejects_escape_and_never_removes_active_run(self):
        state = self.service.store.create("data.csv", ".csv")
        run_id = state["application_run_id"]
        future = datetime.now(timezone.utc) + timedelta(days=2)
        self.assertEqual(self.service.store.cleanup_expired(1, now=future), [])
        self.assertTrue(self.service.store.paths(run_id)["run"].is_dir())
        with self.assertRaises(ServiceError) as context:
            self.service.store.remove("../outside")
        self.assertEqual(context.exception.code, "malformed_run_id")

    def test_cleanup_removes_only_expired_final_run(self):
        state = self.service.store.create("data.csv", ".csv")
        run_id = state["application_run_id"]
        self.service.store.update(run_id, state="failed")
        future = datetime.now(timezone.utc) + timedelta(days=2)
        removed = self.service.store.cleanup_expired(1, now=future)
        self.assertEqual(removed, [run_id])
        with self.assertRaises(ServiceError) as context:
            self.service.store.paths(run_id)
        self.assertEqual(context.exception.code, "run_not_found")


if __name__ == "__main__":
    unittest.main()
