"""HTTP coverage for Phase 10 deterministic conversations."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.dataset_resolver import ResolvedDataset
from api.run_service import ApplicationRunService
from api.server import create_app
from app.config import ApplicationSettings
from tests.api.test_application_flow import _write_successful_pipeline1_artifacts


@pytest.fixture
def dataframe() -> pd.DataFrame:
    return pd.DataFrame({
        "sales": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "revenue": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "salary": [50.0, 70.0, 60.0, 80.0, 90.0, 100.0],
        "region": ["North", "North", "South", "South", "East", "East"],
        "department": ["A", "A", "B", "B", "C", "C"],
        "product": ["P1", "P2", "P3", "P4", "P5", "P6"],
    })


@pytest.fixture
def client(tmp_path: Path):
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
    with TestClient(create_app(service=service)) as test_client:
        yield test_client
    service.shutdown(wait=True)


def _dataset(dataframe: pd.DataFrame) -> ResolvedDataset:
    return ResolvedDataset(dataframe, None, "api_test", Path("cleaned.csv"))


def test_create_visualization_conversation(client, dataframe) -> None:
    with patch("api.routes.conversations.resolve_cleaned_dataset", return_value=_dataset(dataframe)):
        response = client.post(
            "/api/conversations/dataset-a", json={"message": "show sales by region"}
        )
    body = response.json()
    assert response.status_code == 200
    assert body["intent"] == "VISUALIZATION"
    assert body["visualization"]["status"] == "READY"
    assert body["insight"] is None
    assert body["conversation_id"]


def test_followup_request_reuses_conversation(client, dataframe) -> None:
    with patch("api.routes.conversations.resolve_cleaned_dataset", return_value=_dataset(dataframe)):
        first = client.post(
            "/api/conversations/dataset-a", json={"message": "show salary by department"}
        ).json()
        response = client.post(
            f"/api/conversations/dataset-a/{first['conversation_id']}",
            json={"message": "use average instead"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["conversation_id"] == first["conversation_id"]
    assert body["visualization"]["visualizations"][0]["chart_data"]["aggregation"] == "mean"


def test_unknown_conversation_id_returns_404(client, dataframe) -> None:
    response = client.post(
        "/api/conversations/dataset-a/not-real",
        json={"message": "use average instead"},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "conversation_not_found"


def test_conversation_wrong_dataset_returns_409(client, dataframe) -> None:
    with patch("api.routes.conversations.resolve_cleaned_dataset", return_value=_dataset(dataframe)):
        first = client.post(
            "/api/conversations/dataset-a", json={"message": "show sales by region"}
        ).json()
        response = client.post(
            f"/api/conversations/dataset-b/{first['conversation_id']}",
            json={"message": "use average instead"},
        )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "conversation_dataset_mismatch"


def test_clarification_continuation_through_api(client) -> None:
    dataframe = pd.DataFrame({
        "gross_sales": [10.0, 20.0, 30.0, 40.0],
        "net_sales": [8.0, 16.0, 24.0, 32.0],
        "region": ["N", "S", "N", "S"],
    })
    with patch("api.routes.conversations.resolve_cleaned_dataset", return_value=_dataset(dataframe)):
        first = client.post(
            "/api/conversations/dataset-a", json={"message": "show sales by region"}
        ).json()
        assert first["status"] == "NEEDS_CLARIFICATION"
        response = client.post(
            f"/api/conversations/dataset-a/{first['conversation_id']}",
            json={"message": "net sales"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "READY"
    assert body["visualization"]["visualizations"][0]["y_column"] == "net_sales"


def test_insight_result_is_passed_through(client, dataframe) -> None:
    with patch("api.routes.conversations.resolve_cleaned_dataset", return_value=_dataset(dataframe)):
        response = client.post(
            "/api/conversations/dataset-a",
            json={"message": "Which region has the highest revenue?"},
        )
    body = response.json()
    assert response.status_code == 200
    assert body["intent"] == "INSIGHT"
    assert body["insight"]["status"] == "READY"
    assert body["visualization"] is None


def test_request_body_is_typed_and_controlled(client) -> None:
    response = client.post(
        "/api/conversations/dataset-a", json={"message": "   "}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_conversation_request"
