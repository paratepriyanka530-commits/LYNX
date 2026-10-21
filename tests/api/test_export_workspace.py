from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
import time
import zipfile

from fastapi.testclient import TestClient
from openpyxl import load_workbook

from api.run_service import ApplicationRunService
from api.server import create_app
from app.config import ApplicationSettings
from tests.api.test_application_flow import _write_successful_pipeline1_artifacts


def _export_runner(input_path, output_directory):
    report = _write_successful_pipeline1_artifacts(input_path, output_directory)
    cleaned = Path(output_directory) / "cleaned_dataset.csv"
    cleaned.write_text("category,amount,note\nTechnology,25,=2+2\nFurniture,,safe\n", encoding="utf-8")
    report["decision_summary"]["privacy_actions"]["cleaned_dataset_sha256"] = hashlib.sha256(cleaned.read_bytes()).hexdigest()
    report["validation"]["after_summary"].update({
        "row_count": 2, "column_count": 3, "total_missing_cells": 1, "duplicate_row_count": 0,
    })
    (Path(output_directory) / "cleaning_report.json").write_text(json.dumps(report), encoding="utf-8")
    return report


def _blocked_runner(input_path, output_directory):
    report = _export_runner(input_path, output_directory)
    report["decision_summary"]["privacy_actions"]["privacy_clearance_passed"] = False
    (Path(output_directory) / "cleaning_report.json").write_text(json.dumps(report), encoding="utf-8")
    return report


class TestExportWorkspace:
    def setup_method(self):
        self.temporary = tempfile.TemporaryDirectory()
        settings = ApplicationSettings(
            runtime_root=Path(self.temporary.name) / "runtime",
            maximum_upload_bytes=1024 * 1024,
            pipeline1_worker_count=1,
            run_retention_seconds=60,
            upload_chunk_bytes=32,
            xlsx_max_entries=100,
            xlsx_max_uncompressed_bytes=1024 * 1024,
            xlsx_max_compression_ratio=100,
        )
        self.service = ApplicationRunService(settings, pipeline1_runner=_export_runner)
        self.client = TestClient(create_app(service=self.service))
        created = self.client.post(
            "/api/runs",
            files={"dataset": ("Sales_Data.csv", b"category,amount,note\nTechnology,25,test\n", "text/csv")},
        )
        self.run_id = created.json()["run_id"]
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if self.client.get(f"/api/runs/{self.run_id}").json()["state"] == "complete":
                break
            time.sleep(.01)

    def teardown_method(self):
        self.service.shutdown(wait=True)
        self.temporary.cleanup()

    def test_cleaned_csv_and_safe_xlsx_are_real_downloads(self):
        csv_response = self.client.get(f"/api/runs/{self.run_id}/exports/dataset?format=csv")
        assert csv_response.status_code == 200
        assert csv_response.content == (self.service.store.paths(self.run_id)["pipeline1"] / "cleaned_dataset.csv").read_bytes()
        assert "Sales_Data_cleaned.csv" in csv_response.headers["content-disposition"]

        xlsx_response = self.client.get(f"/api/runs/{self.run_id}/exports/dataset?format=xlsx")
        assert xlsx_response.status_code == 200
        workbook = load_workbook(BytesIO(xlsx_response.content), read_only=True)
        rows = list(workbook.active.iter_rows(values_only=True))
        assert rows[0] == ("category", "amount", "note")
        assert rows[1][2] == "'=2+2"
        assert rows[2][1] is None

    def test_availability_and_overview_reports_use_completed_artifacts(self):
        availability = self.client.get(f"/api/runs/{self.run_id}/exports/availability")
        assert availability.status_code == 200
        states = {item["key"]: item["status"] for item in availability.json()["sections"]}
        assert states["overview"] == "available"
        assert states["prediction"] == "unavailable"
        assert states["insights"] == "unavailable"

        pdf = self.client.post(
            f"/api/runs/{self.run_id}/exports/report",
            json={"format": "pdf", "sections": ["overview"]},
        )
        assert pdf.status_code == 200
        assert pdf.content.startswith(b"%PDF")
        assert pdf.headers["content-type"].startswith("application/pdf")

        docx = self.client.post(
            f"/api/runs/{self.run_id}/exports/report",
            json={"format": "docx", "sections": ["overview"]},
        )
        assert docx.status_code == 200
        with zipfile.ZipFile(BytesIO(docx.content)) as package:
            document = package.read("word/document.xml").decode("utf-8")
            assert "Overview" in document
            assert "Prepared rows: 2" in document

    def test_unavailable_section_is_rejected_instead_of_fabricated(self):
        response = self.client.post(
            f"/api/runs/{self.run_id}/exports/report",
            json={"format": "pdf", "sections": ["prediction"]},
        )
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "report_section_unavailable"


def test_privacy_failure_blocks_all_export_downloads(tmp_path):
    settings = ApplicationSettings(
        runtime_root=tmp_path / "runtime", maximum_upload_bytes=1024 * 1024,
        pipeline1_worker_count=1, run_retention_seconds=60, upload_chunk_bytes=32,
        xlsx_max_entries=100, xlsx_max_uncompressed_bytes=1024 * 1024,
        xlsx_max_compression_ratio=100,
    )
    service = ApplicationRunService(settings, pipeline1_runner=_blocked_runner)
    try:
        with TestClient(create_app(service=service)) as client:
            created = client.post("/api/runs", files={"dataset": ("blocked.csv", b"a,b\n1,2\n", "text/csv")})
            run_id = created.json()["run_id"]
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                if client.get(f"/api/runs/{run_id}").json()["state"] == "complete": break
                time.sleep(.01)
            response = client.get(f"/api/runs/{run_id}/exports/dataset?format=csv")
            assert response.status_code == 403
            assert response.json()["error"]["code"] == "privacy_clearance_not_passed"
    finally:
        service.shutdown(wait=True)
