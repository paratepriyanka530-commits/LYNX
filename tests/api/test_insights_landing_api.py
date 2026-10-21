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
from tests.pipeline3_results.insights.landing.test_summary import _report


@pytest.fixture
def landing_client(tmp_path: Path):
    settings = ApplicationSettings(runtime_root=tmp_path / "runtime")
    service = ApplicationRunService(
        settings, pipeline1_runner=_write_successful_pipeline1_artifacts,
    )
    with TestClient(create_app(service=service)) as client:
        yield client
    service.shutdown(wait=True)


def _resolved(tmp_path: Path) -> ResolvedDataset:
    dataframe = pd.DataFrame({
        "amount":[10.0,20.0,30.0],
        "category":["A","A","B"],
        "date":["2025-01-01","2025-02-01","2025-03-01"],
    })
    path = tmp_path / "cleaned.csv"
    path.write_text("amount,category,date\n10,A,2025-01-01\n", encoding="utf-8")
    return ResolvedDataset(
        dataframe,
        _report(dataframe, measures=("amount",), dimensions=("category",), temporal=("date",)),
        "retail_data",
        path,
    )


def test_landing_endpoint_returns_additive_summary_contract(landing_client, tmp_path) -> None:
    with patch("api.routes.insights.resolve_cleaned_dataset", return_value=_resolved(tmp_path)):
        response = landing_client.get("/api/insights/dataset-a/landing")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "READY"
    assert body["summary_text"]
    assert 1 <= len(body["suggested_questions"]) <= 5
    assert all(set(item) == {"id", "text", "analytical_goal"} for item in body["suggested_questions"])


def test_landing_endpoint_is_cached_by_dataset_version(landing_client, tmp_path) -> None:
    resolved = _resolved(tmp_path)
    with patch("api.routes.insights.resolve_cleaned_dataset", return_value=resolved):
        first = landing_client.get("/api/insights/dataset-a/landing")
        second = landing_client.get("/api/insights/dataset-a/landing")
    assert first.status_code == second.status_code == 200
    assert "insights_landing:cache_hit" in second.json()["signals"]


def test_existing_query_endpoint_shape_is_unchanged(landing_client, tmp_path) -> None:
    resolved = _resolved(tmp_path)
    with patch("api.routes.insights.resolve_cleaned_dataset", return_value=resolved):
        response = landing_client.post(
            "/api/insights/dataset-a/query",
            json={"request_text":"What is the average amount?"},
        )
    assert response.status_code == 200
    assert set(response.json()) == {
        "status", "answer", "result", "evidence", "message",
        "clarification_options", "errors", "signals", "failure_code",
    }
