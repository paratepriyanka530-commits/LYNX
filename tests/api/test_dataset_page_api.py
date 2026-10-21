from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import time

import pandas as pd
from fastapi.testclient import TestClient

from api.run_service import ApplicationRunService
from api.server import create_app
from app.config import ApplicationSettings
from tests.api.test_application_flow import _write_successful_pipeline1_artifacts


def _cleaning_runner(input_path, output_directory):
    report = _write_successful_pipeline1_artifacts(
        input_path, output_directory, pii_columns=("customer",)
    )
    cleaned_path = Path(output_directory) / "cleaned_dataset.csv"
    frame = pd.read_csv(cleaned_path)
    frame["category"] = frame["category"].astype(str).str.strip().str.title()
    frame.to_csv(cleaned_path, index=False)
    report["decision_summary"]["privacy_actions"]["cleaned_dataset_sha256"] = (
        hashlib.sha256(cleaned_path.read_bytes()).hexdigest()
    )
    (Path(output_directory) / "cleaning_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    return report


class TestDatasetPageApi:
    def setup_method(self):
        self.temporary = tempfile.TemporaryDirectory()
        settings = ApplicationSettings(
            runtime_root=Path(self.temporary.name) / "runtime",
            maximum_upload_bytes=1024 * 1024,
            pipeline1_worker_count=1,
            run_retention_seconds=60,
            upload_chunk_bytes=7,
            xlsx_max_entries=100,
            xlsx_max_uncompressed_bytes=1024 * 1024,
            xlsx_max_compression_ratio=100,
        )
        self.service = ApplicationRunService(settings, pipeline1_runner=_cleaning_runner)
        self.client = TestClient(create_app(service=self.service))
        content = (
            b"category,region,amount,customer\n"
            b" technology ,East,25,Alice\n"
            b"Furniture,West,40,Bob\n"
            b" technology ,Central,15,Carol\n"
            b"Office Supplies,East,10,Dan\n"
            b"Furniture,East,55,Eve\n"
        )
        response = self.client.post(
            "/api/runs", files={"dataset": ("Sales_Data.csv", content, "text/csv")}
        )
        self.run_id = response.json()["run_id"]
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if self.client.get(f"/api/runs/{self.run_id}").json()["state"] == "complete":
                break
            time.sleep(0.01)

    def teardown_method(self):
        self.service.shutdown(wait=True)
        self.temporary.cleanup()

    def test_original_and_cleaned_versions_use_their_authoritative_artifacts(self):
        original = self.client.get(
            f"/api/runs/{self.run_id}/dataset?version=original&limit=20"
        )
        cleaned = self.client.get(
            f"/api/runs/{self.run_id}/dataset?version=cleaned&limit=20"
        )
        assert original.status_code == cleaned.status_code == 200
        assert original.json()["rows"][0][0] == " technology "
        assert cleaned.json()["rows"][0][0] == "Technology"
        assert original.json()["privacy_redacted_columns"] == ["customer"]
        assert original.json()["rows"][0][3] != "Alice"
        assert cleaned.json()["change_highlighting_available"] is False

    def test_pagination_is_bounded_and_reports_filtered_totals(self):
        page = self.client.get(
            f"/api/runs/{self.run_id}/dataset?version=cleaned&offset=2&limit=2"
        )
        assert page.status_code == 200
        assert len(page.json()["rows"]) == 2
        assert page.json()["row_numbers"] == [3, 4]
        assert page.json()["total_rows"] == 5
        assert page.json()["filtered_rows"] == 5

    def test_search_filter_and_sort_are_applied_before_pagination(self):
        response = self.client.get(
            f"/api/runs/{self.run_id}/dataset",
            params={
                "version": "cleaned", "search": "east", "filter_column": "region",
                "filter_value": "east", "sort_column": "amount", "sort_direction": "desc",
                "limit": 20,
            },
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["filtered_rows"] == 3
        assert [row[2] for row in payload["rows"]] == [55, 25, 10]

    def test_invalid_columns_return_controlled_errors(self):
        response = self.client.get(
            f"/api/runs/{self.run_id}/dataset?version=cleaned&sort_column=missing"
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_sort_column"


def test_failed_privacy_clearance_withholds_dataset_rows(tmp_path: Path):
    settings = ApplicationSettings(runtime_root=tmp_path / "runtime")
    service = ApplicationRunService(
        settings,
        pipeline1_runner=lambda source, output: _write_successful_pipeline1_artifacts(
            source, output, privacy_clear=False
        ),
    )
    client = TestClient(create_app(service=service))
    try:
        response = client.post(
            "/api/runs", files={"dataset": ("private.csv", b"name\nAlice\n", "text/csv")}
        )
        run_id = response.json()["run_id"]
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if client.get(f"/api/runs/{run_id}").json()["state"] == "complete":
                break
            time.sleep(0.01)
        result = client.get(f"/api/runs/{run_id}/dataset?version=original")
        assert result.status_code == 403
        assert result.json()["error"]["code"] == "privacy_clearance_not_passed"
    finally:
        service.shutdown(wait=True)
