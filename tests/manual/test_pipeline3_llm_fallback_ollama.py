"""Opt-in local Ollama smoke test for Phase 11.

Run manually with:
    $env:LYNX_RUN_OLLAMA_SMOKE="1"
    python -m pytest tests/manual/test_pipeline3_llm_fallback_ollama.py -s -q
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from intelligence.llm.ollama_provider import OllamaProvider
from pipelines.pipeline3_results.insights.results_service import build_insight_results
from pipelines.pipeline3_results.visualization.results_service import (
    build_visualization_results,
)


pytestmark = pytest.mark.skipif(
    os.getenv("LYNX_RUN_OLLAMA_SMOKE") != "1",
    reason="manual Ollama smoke test is opt-in",
)


class RecordingOllamaProvider(OllamaProvider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.structured_outputs = []

    def generate_structured(self, prompt, response_model):
        output = super().generate_structured(prompt, response_model)
        self.structured_outputs.append(output)
        return output


def test_real_qwen_semantic_fallback_observations() -> None:
    dataframe = pd.DataFrame({
        "salary": [50.0, 70.0, 60.0, 80.0],
        "department": ["A", "A", "B", "B"],
        "sales": [10.0, 20.0, 30.0, 40.0],
        "revenue": [12.0, 22.0, 32.0, 42.0],
        "region": ["North", "South", "North", "South"],
        "month": pd.to_datetime(["2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01"]),
        "gross_sales": [11.0, 21.0, 31.0, 41.0],
        "net_sales": [9.0, 19.0, 29.0, 39.0],
    })
    provider = RecordingOllamaProvider(model="qwen2.5:7b")
    cases = [
        (
            "give me a breakdown of what people earn across each department",
            build_visualization_results,
        ),
        ("Which area brought in the most money?", build_insight_results),
        (
            "I'd like to see how revenue changed as the months went by.",
            build_visualization_results,
        ),
    ]
    for phrase, service in cases:
        output_count = len(provider.structured_outputs)
        if service is build_visualization_results:
            result = service(dataframe, phrase, llm_provider=provider)
        else:
            result = service(dataframe, phrase, llm_provider=provider)
        print({
            "raw_phrase": phrase,
            "structured_interpretation": (
                provider.structured_outputs[-1]
                if len(provider.structured_outputs) > output_count
                else "rejected_by_strict_schema"
            ),
            "status": result["status"],
            "signals": [s for s in result.get("signals", []) if s.startswith("llm:")],
        })
        assert result["status"] in {"READY", "NEEDS_CLARIFICATION", "INVALID"}

    ambiguous = build_visualization_results(
        dataframe[["gross_sales", "net_sales", "region"]],
        "show sales by region",
        llm_provider=provider,
    )
    print({
        "raw_phrase": "show sales by region",
        "status": ambiguous["status"],
        "clarification_options": ambiguous.get("clarification_options", []),
    })
    assert ambiguous["status"] == "NEEDS_CLARIFICATION"
    assert set(ambiguous["clarification_options"]) == {"gross_sales", "net_sales"}
