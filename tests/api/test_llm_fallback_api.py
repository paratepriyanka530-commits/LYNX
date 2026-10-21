"""API wiring checks for the optional Phase 11 semantic provider."""

from __future__ import annotations

from pathlib import Path
from typing import Type
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient
from pydantic import BaseModel

from api.dataset_resolver import ResolvedDataset
from api.run_service import ApplicationRunService
from api.server import create_app
from app.config import ApplicationSettings
from intelligence.llm.provider import LLMProvider
from tests.api.test_application_flow import _write_successful_pipeline1_artifacts


class APIProvider(LLMProvider):
    def __init__(self):
        self.calls = 0

    def generate_structured(self, prompt: str, response_model: Type[BaseModel]):
        self.calls += 1
        measure = "sales" if "show sales by department" in prompt else "salary"
        return {
            "intent": "VISUALIZATION",
            "analytical_goal": "GROUP_COMPARISON",
            "measure_concepts": [measure],
            "dimension_concepts": ["department"],
            "aggregation": "MEAN" if measure == "salary" else "SUM",
            "comparison": "NONE",
            "limit": None,
            "requested_chart": None,
            "filters": [],
            "time_expression": None,
            "missing_slots": [],
            "ambiguities": [],
            "needs_clarification": False,
            "clarification_question": None,
            "unsupported": False,
            "unsupported_reason": None,
        }


def test_injected_provider_reaches_single_turn_and_conversation_api(tmp_path) -> None:
    dataframe = pd.DataFrame({
        "salary": [50.0, 70.0, 60.0, 80.0],
        "sales": [10.0, 20.0, 30.0, 40.0],
        "department": ["A", "A", "B", "B"],
    })
    resolved = ResolvedDataset(dataframe, None, "test", Path("cleaned.csv"))
    provider = APIProvider()
    settings = ApplicationSettings(runtime_root=tmp_path / "runtime")
    service = ApplicationRunService(
        settings, pipeline1_runner=_write_successful_pipeline1_artifacts
    )
    try:
        with (
            TestClient(create_app(
                service=service, semantic_llm_provider=provider
            )) as client,
            patch(
                "api.routes.visualization.resolve_cleaned_dataset",
                return_value=resolved,
            ),
            patch(
                "api.routes.conversations.resolve_cleaned_dataset",
                return_value=resolved,
            ),
        ):
            visual = client.post(
                "/api/visualizations/dataset-a/query",
                json={
                    "request_text": "give me a breakdown of what people earn across each department"
                },
            )
            assert visual.status_code == 200
            assert visual.json()["status"] == "READY"

            first = client.post(
                "/api/conversations/dataset-a",
                json={"message": "show sales by department"},
            ).json()
            followup = client.post(
                f"/api/conversations/dataset-a/{first['conversation_id']}",
                json={"message": "could you base that on what they earn instead?"},
            )
            assert followup.status_code == 200
            assert followup.json()["status"] == "READY"
            assert followup.json()["visualization"]["visualizations"][0]["y_column"] == "salary"
        assert provider.calls == 3
    finally:
        service.shutdown(wait=True)
