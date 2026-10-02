"""API and UUID-chain coverage for deterministic Pipeline 3 Insights."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.dataset_resolver import ResolvedDataset
from api.run_service import ApplicationRunService
from api.server import create_app
from app.config import ApplicationSettings
from tests.api.test_application_flow import _write_successful_pipeline1_artifacts


@pytest.fixture
def insight_df() -> pd.DataFrame:
    return pd.DataFrame({
        "salary": [50.0, 70.0, 60.0, 80.0, None, 90.0, 55.0, 75.0],
        "revenue": [100.0, 200.0, 50.0, 300.0, 150.0, 250.0, 125.0, 175.0],
        "department": ["A", "A", "B", "B", "C", "C", "D", "D"],
        "region": ["East", "East", "West", "West", "North", "North", "South", "South"],
        "product": ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"],
    })


@pytest.fixture
def api_client(tmp_path: Path):
    settings = ApplicationSettings(
        runtime_root=tmp_path / "runtime",
        maximum_upload_bytes=1024 * 1024,
        pipeline1_worker_count=1,
        run_retention_seconds=60,
        upload_chunk_bytes=1024,
    )
    service = ApplicationRunService(
        settings, pipeline1_runner=_write_successful_pipeline1_artifacts
    )
    with TestClient(create_app(service=service)) as client:
        yield client
    service.shutdown(wait=True)


def _resolved(dataframe: pd.DataFrame, report=None) -> ResolvedDataset:
    return ResolvedDataset(dataframe, report, "test_dataset", Path("cleaned.csv"))


def _post_with_dataframe(client, dataframe, question, report=None):
    with patch(
        "api.routes.insights.resolve_cleaned_dataset",
        return_value=_resolved(dataframe, report),
    ):
        return client.post(
            "/api/insights/known/query", json={"request_text": question}
        )


def test_known_production_dataset_returns_ready(api_client) -> None:
    response = api_client.post(
        "/api/insights/retail/query",
        json={"request_text": "What is the average total amount?"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "READY"
    assert response.json()["answer"]


def test_average_salary_returns_computed_answer(api_client, insight_df) -> None:
    response = _post_with_dataframe(
        api_client, insight_df, "What is the average salary?"
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "READY"
    assert body["result"]["value"] == pytest.approx(480.0 / 7.0)


def test_grouped_comparison_returns_ready(api_client, insight_df) -> None:
    response = _post_with_dataframe(
        api_client,
        insight_df,
        "Which department has the highest average salary?",
    )
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "READY"
    assert body["result"]["winner"] == "C"


def test_top_five_returns_ranked_rows(api_client, insight_df) -> None:
    response = _post_with_dataframe(
        api_client, insight_df, "What are the top 5 products by revenue?"
    )
    body = response.json()
    assert body["status"] == "READY"
    assert [row["group"] for row in body["result"]["rows"]] == [
        "P4", "P6", "P2", "P8", "P5"
    ]


def test_ambiguous_concept_returns_clarification_options(api_client) -> None:
    dataframe = pd.DataFrame({
        "gross_sales": [10.0, 20.0],
        "net_sales": [8.0, 16.0],
    })
    response = _post_with_dataframe(api_client, dataframe, "What is total sales?")
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "NEEDS_CLARIFICATION"
    assert body["clarification_options"] == ["gross_sales", "net_sales"]


def test_missing_concept_is_analytical_invalid(api_client, insight_df) -> None:
    response = _post_with_dataframe(
        api_client, insight_df, "What is average rainfall?"
    )
    assert response.status_code == 200
    assert response.json()["status"] == "INVALID"


def test_vague_request_is_controlled_clarification(api_client, insight_df) -> None:
    response = _post_with_dataframe(api_client, insight_df, "tell me something")
    assert response.status_code == 200
    assert response.json()["status"] == "NEEDS_CLARIFICATION"


def test_visualization_request_is_not_executed(api_client, insight_df) -> None:
    response = _post_with_dataframe(
        api_client, insight_df, "show salary by department"
    )
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "NEEDS_CLARIFICATION"
    assert "visualization request" in body["message"].lower()


def test_unknown_run_uuid_returns_dataset_not_found(api_client) -> None:
    response = api_client.post(
        f"/api/insights/{uuid4()}/query",
        json={"request_text": "How many records are there?"},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "dataset_not_found"


@pytest.mark.parametrize(
    "payload",
    [{}, {"request_text": 42}, {"request_text": "   "}],
)
def test_malformed_typed_request_is_validation_error(api_client, payload) -> None:
    response = api_client.post("/api/insights/retail/query", json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_insight_request"


def test_invalid_json_is_validation_error(api_client) -> None:
    response = api_client.post(
        "/api/insights/retail/query",
        content=b"{not-json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_insight_request"


def test_response_is_json_serializable_and_contract_complete(api_client, insight_df) -> None:
    response = _post_with_dataframe(api_client, insight_df, "What is total revenue?")
    body = response.json()
    assert json.loads(json.dumps(body)) == body
    assert set(body) == {
        "status", "answer", "result", "evidence", "message",
        "clarification_options", "errors", "signals", "failure_code",
    }


def test_service_signals_are_preserved(api_client, insight_df) -> None:
    response = _post_with_dataframe(api_client, insight_df, "What is total revenue?")
    assert "insight_service:analysis_executed" in response.json()["signals"]


def test_input_dataframe_is_not_mutated(api_client, insight_df) -> None:
    before = insight_df.copy(deep=True)
    _post_with_dataframe(
        api_client, insight_df, "Which department has the highest average salary?"
    )
    pd.testing.assert_frame_equal(insight_df, before)


def test_unexpected_failure_uses_controlled_service_error(api_client, insight_df) -> None:
    with (
        patch(
            "api.routes.insights.resolve_cleaned_dataset",
            return_value=_resolved(insight_df),
        ),
        patch(
            "api.routes.insights.build_insight_results",
            side_effect=RuntimeError("private stack detail"),
        ),
    ):
        response = api_client.post(
            "/api/insights/known/query",
            json={"request_text": "What is total revenue?"},
        )
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "insight_generation_error"
    assert "private stack detail" not in response.text


def _semantic_artifact_writer(input_path, output_directory):
    report = _write_successful_pipeline1_artifacts(input_path, output_directory)
    report["semantic_profile"] = {"columns": [
        {
            "column_name": "amt",
            "semantic_type": "numeric_measurement",
            "concept_description": "Annual employee salary",
        },
        {
            "column_name": "dept_cd",
            "semantic_type": "categorical",
            "concept_description": "Employee department name",
        },
    ]}
    (Path(output_directory) / "cleaning_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def test_run_pipeline1_artifacts_to_semantic_insight_chain(tmp_path: Path) -> None:
    settings = ApplicationSettings(
        runtime_root=tmp_path / "custom-runtime-root",
        maximum_upload_bytes=1024 * 1024,
        pipeline1_worker_count=1,
        run_retention_seconds=60,
        upload_chunk_bytes=1024,
    )
    service = ApplicationRunService(settings, pipeline1_runner=_semantic_artifact_writer)
    csv_data = b"amt,dept_cd\n50,A\n70,A\n60,B\n80,B\n90,C\n100,C\n"
    try:
        with TestClient(create_app(service=service)) as client:
            upload = client.post(
                "/api/runs",
                files={"dataset": ("opaque.csv", csv_data, "text/csv")},
            )
            assert upload.status_code == 202
            run_id = upload.json()["run_id"]

            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                status = client.get(f"/api/runs/{run_id}").json()
                if status["state"] == "complete":
                    break
                assert status["state"] != "failed", status
                time.sleep(0.02)
            else:
                pytest.fail("Pipeline 1 artifact writer did not complete")

            response = client.post(
                f"/api/insights/{run_id}/query",
                json={
                    "request_text": "Which department has the highest average salary?"
                },
            )
            body = response.json()
            assert response.status_code == 200
            assert body["status"] == "READY"
            assert body["result"]["measure_column"] == "amt"
            assert body["result"]["dimension_column"] == "dept_cd"
            assert body["result"]["winner"] == "C"
            assert (settings.runtime_root / run_id / "pipeline1" / "cleaning_report.json").is_file()
    finally:
        service.shutdown(wait=True)
