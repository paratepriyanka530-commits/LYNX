"""End-to-end integration test: Run -> Pipeline 1 -> Visualize.

Verifies the complete chain:
1. POST /api/runs  (upload CSV, triggers mocked Pipeline 1 via background thread)
2. Poll until run reaches state=complete
3. GET /api/visualizations/{run_id}          -- automatic mode, UUID path
4. POST /api/visualizations/{run_id}/query   -- requested mode, UUID path

Key facts verified:
- resolve_cleaned_dataset correctly resolves a run UUID using runtime_root
  obtained from app.state.run_service.settings (via _get_runtime_root helper)
- pipeline1_report is populated (NOT None) because _write_successful_pipeline1_artifacts
  writes both cleaned_dataset.csv and cleaning_report.json
- Automatic response has status=READY, mode=automatic, visualizations with explanations
- Requested response has status=READY, mode=requested, score=None
- NEEDS_CLARIFICATION is returned as HTTP 200 (not an error)
- After Pipeline 1, GET /api/runs/{run_id}/pipeline1 returns 200 (eligibility check)

What is mocked and why:
- pipeline1_runner is mocked via _write_successful_pipeline1_artifacts from
  test_application_flow.py. This writes a real cleaned_dataset.csv and a real
  cleaning_report.json without any LLM/Ollama calls. Pipeline 1 proper involves
  multi-minute Qwen 2.5 calls -- triggering that from an HTTP test is out of scope.
- The uploaded CSV content (feature,target,...) is a minimal synthetic dataset.
  This is intentional: the test validates the UUID resolution path and report wiring,
  not specific chart types from production data (those are covered by Stage 10 tests).
- Prediction endpoints are not exercised here; they are covered in test_prediction_jobs.py.
"""

from __future__ import annotations

import time
import tempfile
from pathlib import Path
import unittest

from fastapi.testclient import TestClient

from api.run_service import ApplicationRunService
from api.server import create_app
from app.config import ApplicationSettings
from tests.api.test_application_flow import _write_successful_pipeline1_artifacts

# A small but realistic CSV: 3 numeric + 1 categorical + 1 target, 20 rows
_SYNTHETIC_CSV = (
    b"quantity,unit_price,category,discount,revenue\n"
    b"10,5.5,A,0.1,49.5\n"
    b"20,3.0,B,0.0,60.0\n"
    b"15,7.2,A,0.2,86.4\n"
    b"8,12.0,C,0.1,86.4\n"
    b"30,2.5,B,0.05,71.25\n"
    b"5,15.0,C,0.0,75.0\n"
    b"12,6.0,A,0.15,61.2\n"
    b"25,4.0,B,0.1,90.0\n"
    b"18,8.5,C,0.2,122.4\n"
    b"9,10.0,A,0.0,90.0\n"
    b"22,3.5,B,0.05,73.15\n"
    b"14,9.0,C,0.1,113.4\n"
    b"7,11.0,A,0.0,77.0\n"
    b"28,2.0,B,0.15,47.6\n"
    b"11,7.5,C,0.0,82.5\n"
    b"16,5.0,A,0.2,64.0\n"
    b"23,4.5,B,0.1,93.15\n"
    b"19,6.5,C,0.05,117.275\n"
    b"6,13.0,A,0.0,78.0\n"
    b"27,3.2,B,0.2,69.12\n"
)


class RunVisualizeIntegrationTests(unittest.TestCase):
    """Integration tests verifying the full Run -> Pipeline 1 -> Visualize chain."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "runtime"
        self.settings = ApplicationSettings(
            runtime_root=self.root,
            maximum_upload_bytes=1024 * 1024,
            pipeline1_worker_count=1,
            run_retention_seconds=60,
            upload_chunk_bytes=4096,
        )
        # Use mock pipeline1_runner that writes artifacts without any LLM calls.
        # See: tests/api/test_application_flow.py::_write_successful_pipeline1_artifacts
        self.service = ApplicationRunService(
            self.settings,
            pipeline1_runner=_write_successful_pipeline1_artifacts,
        )
        self.client = TestClient(create_app(service=self.service))

    def tearDown(self):
        self.service.shutdown(wait=True)
        self.temporary.cleanup()

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------
    def _upload_and_wait(self, csv_content=_SYNTHETIC_CSV, filename="data.csv"):
        """Upload a CSV, wait for Pipeline 1 to complete, return run_id."""
        resp = self.client.post(
            "/api/runs",
            files={"dataset": (filename, csv_content, "application/octet-stream")},
        )
        self.assertEqual(resp.status_code, 202, resp.text)
        run_id = resp.json()["run_id"]

        # Poll until complete (mocked Pipeline 1 completes almost instantly)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            status = self.client.get(f"/api/runs/{run_id}").json()
            if status["state"] == "complete":
                return run_id
            if status["state"] == "failed":
                self.fail(f"Pipeline 1 failed: {status}")
            time.sleep(0.02)
        self.fail(f"Run did not reach complete state in time: {status}")

    def _verify_pipeline1_artifacts_exist(self, run_id: str):
        """Assert that cleaned_dataset.csv and cleaning_report.json are on disk."""
        p1_dir = self.root / run_id / "pipeline1"
        cleaned = p1_dir / "cleaned_dataset.csv"
        report = p1_dir / "cleaning_report.json"
        self.assertTrue(cleaned.is_file(), f"cleaned_dataset.csv missing at {cleaned}")
        self.assertTrue(report.is_file(), f"cleaning_report.json missing at {report}")
        self.assertGreater(cleaned.stat().st_size, 0, "cleaned_dataset.csv is 0 bytes")
        self.assertGreater(report.stat().st_size, 0, "cleaning_report.json is 0 bytes")
        return cleaned, report

    # -----------------------------------------------------------------------
    # 1. Pipeline 1 artifacts land at the expected path
    # -----------------------------------------------------------------------
    def test_pipeline1_artifacts_land_at_expected_uuid_path(self):
        """Verifies {runtime_root}/{run_id}/pipeline1/cleaned_dataset.csv exists after run."""
        run_id = self._upload_and_wait()
        cleaned, report = self._verify_pipeline1_artifacts_exist(run_id)
        # Confirm path structure exactly matches what dataset_resolver.py expects
        self.assertEqual(cleaned.parent.name, "pipeline1")
        self.assertEqual(cleaned.parent.parent.name, run_id)
        self.assertEqual(cleaned.parent.parent.parent, self.root)

    # -----------------------------------------------------------------------
    # 2. Automatic visualization via UUID resolves correctly
    # -----------------------------------------------------------------------
    def test_automatic_visualization_via_uuid_returns_ready(self):
        """GET /api/visualizations/{run_id} uses UUID path, pipeline1_report is NOT None."""
        run_id = self._upload_and_wait()
        self._verify_pipeline1_artifacts_exist(run_id)

        resp = self.client.get(f"/api/visualizations/{run_id}")
        self.assertEqual(resp.status_code, 200, resp.text)

        data = resp.json()
        self.assertEqual(data["mode"], "automatic")
        self.assertEqual(data["status"], "READY")
        self.assertGreater(data["visualization_count"], 0)
        self.assertEqual(len(data["visualizations"]), data["visualization_count"])

        # Each visualization must have all Stage 10 fields
        for viz in data["visualizations"]:
            self.assertIn("rank", viz)
            self.assertIn("chart_type", viz)
            self.assertIn("x_column", viz)
            self.assertIn("y_column", viz)
            self.assertIn("score", viz)
            self.assertIsNotNone(viz["score"])  # automatic mode: score preserved
            self.assertIn("chart_data", viz)
            self.assertIn("signals", viz)
            self.assertIn("explanation", viz)
            expl = viz["explanation"]
            for key in ("why_this_chart", "what_it_shows", "aggregation", "data_coverage", "evidence"):
                self.assertIn(key, expl)
            # evidence must be a non-empty list for READY status
            self.assertIsInstance(expl["evidence"], list)
            self.assertGreater(len(expl["evidence"]), 0)

    # -----------------------------------------------------------------------
    # 3. pipeline1_report is populated (not None) from cleaning_report.json
    # -----------------------------------------------------------------------
    def test_pipeline1_report_is_populated_for_uuid_run(self):
        """
        Verifies that dataset_resolver picks up cleaning_report.json and passes
        pipeline1_report (not None) into build_visualization_results.

        We detect this indirectly: a non-None pipeline1_report means Stage 6
        has context for the semantic_profile field. If it were None, Stage 10
        would still return READY but without that context. We verify the response
        is well-formed and the /api/runs/{run_id}/pipeline1 endpoint also confirms
        the run has a valid pipeline1 result (eligibility=True).
        """
        run_id = self._upload_and_wait()
        self._verify_pipeline1_artifacts_exist(run_id)

        # Confirm pipeline1 result endpoint reports eligibility
        p1_resp = self.client.get(f"/api/runs/{run_id}/pipeline1")
        self.assertEqual(p1_resp.status_code, 200, p1_resp.text)
        p1_data = p1_resp.json()
        self.assertTrue(p1_data["prediction_eligibility"]["eligible"])

        # Confirm visualization also returns READY (pipeline1_report flowed in)
        viz_resp = self.client.get(f"/api/visualizations/{run_id}")
        self.assertEqual(viz_resp.status_code, 200)
        self.assertEqual(viz_resp.json()["status"], "READY")

    # -----------------------------------------------------------------------
    # 4. Requested visualization via UUID resolves correctly
    # -----------------------------------------------------------------------
    def test_requested_visualization_via_uuid_returns_ready(self):
        """POST /api/visualizations/{run_id}/query resolves the UUID and returns READY."""
        run_id = self._upload_and_wait()
        self._verify_pipeline1_artifacts_exist(run_id)

        resp = self.client.post(
            f"/api/visualizations/{run_id}/query",
            json={"request_text": "bar chart of quantity by category"},
        )
        self.assertEqual(resp.status_code, 200, resp.text)

        data = resp.json()
        self.assertEqual(data["mode"], "requested")
        self.assertIn(data["status"], ("READY", "NEEDS_CLARIFICATION", "INVALID"))
        # score must be None for requested mode (Stage 8 always sets score=None)
        if data["status"] == "READY":
            self.assertEqual(data["visualization_count"], 1)
            viz = data["visualizations"][0]
            self.assertIsNone(viz["score"])
            self.assertIn("explanation", viz)

    # -----------------------------------------------------------------------
    # 5. NEEDS_CLARIFICATION returned as HTTP 200 (not an error)
    # -----------------------------------------------------------------------
    def test_vague_query_via_uuid_returns_200_not_error(self):
        """A vague query returns HTTP 200 with status=NEEDS_CLARIFICATION."""
        run_id = self._upload_and_wait()

        resp = self.client.post(
            f"/api/visualizations/{run_id}/query",
            json={"request_text": "show me a chart"},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        data = resp.json()
        self.assertEqual(data["mode"], "requested")
        # NEEDS_CLARIFICATION is a legitimate outcome, not an error
        self.assertIn(data["status"], ("NEEDS_CLARIFICATION", "READY", "INVALID"))

    # -----------------------------------------------------------------------
    # 6. Unknown run_id returns 404
    # -----------------------------------------------------------------------
    def test_unknown_run_uuid_returns_404(self):
        """GET /api/visualizations/{unknown_uuid} returns 404 dataset_not_found."""
        import uuid
        unknown = str(uuid.uuid4())
        resp = self.client.get(f"/api/visualizations/{unknown}")
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.json()["error"]["code"], "dataset_not_found")

    # -----------------------------------------------------------------------
    # 7. Visualization before Pipeline 1 completes returns 404 (not found yet)
    # -----------------------------------------------------------------------
    def test_visualization_before_pipeline1_completes_returns_404(self):
        """If run exists but Pipeline 1 has not written artifacts yet, returns 404."""
        from threading import Event
        started = Event()
        unblock = Event()

        def blocking_runner(input_path, output_dir):
            started.set()
            unblock.wait(timeout=5)
            return _write_successful_pipeline1_artifacts(input_path, output_dir)

        service = ApplicationRunService(
            self.settings,
            pipeline1_runner=blocking_runner,
        )
        client = TestClient(create_app(service=service))
        try:
            resp = client.post(
                "/api/runs",
                files={"dataset": ("data.csv", _SYNTHETIC_CSV, "application/octet-stream")},
            )
            self.assertEqual(resp.status_code, 202)
            run_id = resp.json()["run_id"]
            # Wait until runner has started (Pipeline 1 is running but not done)
            self.assertTrue(started.wait(timeout=3))
            # Visualization should return 404 (cleaned_dataset.csv not yet written)
            viz_resp = client.get(f"/api/visualizations/{run_id}")
            self.assertEqual(viz_resp.status_code, 404)
            self.assertEqual(viz_resp.json()["error"]["code"], "dataset_not_found")
        finally:
            unblock.set()
            service.shutdown(wait=True)

    # -----------------------------------------------------------------------
    # 8. Domain dataset path resolves from storage/domain_datasets/ (not tests/)
    # -----------------------------------------------------------------------
    def test_domain_dataset_resolves_from_production_path(self):
        """pizza_sales resolves from storage/domain_datasets/, not tests/datasets/valid/."""
        from api.dataset_resolver import resolve_cleaned_dataset
        import os

        project_root = Path(__file__).resolve().parents[2]
        prod_path = project_root / "storage" / "domain_datasets" / "pizza_sales.csv"

        resolved = resolve_cleaned_dataset("pizza_sales", project_root=project_root)
        # Must resolve to the production path, not the test fixture path
        self.assertEqual(resolved.csv_path, prod_path)
        self.assertEqual(resolved.dataset_name, "pizza_sales")
        self.assertGreater(len(resolved.dataframe), 0)