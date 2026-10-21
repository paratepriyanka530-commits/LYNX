from __future__ import annotations

import pandas as pd
import pytest

from intelligence.llm.provider import LLMProvider
from pipelines.pipeline3_results.execution.pandas_executor import PandasAnalyticalExecutor
from pipelines.pipeline3_results.grounding.dataset_knowledge import build_dataset_knowledge
from pipelines.pipeline3_results.grounding.schemas import AnalyticalExpression
from pipelines.pipeline3_results.insights.landing.opportunities import make_plan
from pipelines.pipeline3_results.visualization.intelligence.schemas import ChartFamily
from pipelines.pipeline3_results.visualization.intelligence.service import build_requested_visualization
from pipelines.pipeline3_results.visualization.results_service import build_visualization_results
from pipelines.pipeline3_results.visualization.xai.schemas import XAIModelDraft
from pipelines.pipeline3_results.visualization.xai.service import VisualizationXAIService


class Provider(LLMProvider):
    def __init__(self, response=None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.prompt = ""

    def generate_structured(self, prompt, response_model):
        assert response_model is XAIModelDraft
        self.prompt = prompt
        if self.error:
            raise self.error
        return self.response


def _scatter(provider=None, language="en"):
    frame = pd.DataFrame({"metric_alpha": [1.0, 2.0, 3.0], "metric_beta": [2.0, 4.0, 6.0], "raw_secret": ["never-send-a", "never-send-b", "never-send-c"]})
    knowledge = build_dataset_knowledge(None, dataset_id="generic", dataframe_schema={str(c): str(t) for c, t in frame.dtypes.items()})
    plan = make_plan("RELATIONSHIP", measures=[AnalyticalExpression(kind="COLUMN", column="metric_alpha"), AnalyticalExpression(kind="COLUMN", column="metric_beta")])
    result = PandasAnalyticalExecutor().execute(plan, frame, knowledge)
    _, visual = build_requested_visualization(plan, result, knowledge, ChartFamily.SCATTER)
    assert visual is not None
    return VisualizationXAIService().build(result, visual.plan, visual.chart_spec, provider=provider, language=language)


def test_deterministic_wording_is_concise_and_distinguishes_rationales() -> None:
    xai = _scatter()
    assert xai.status == "READY" and xai.fallback_used is True
    assert "numeric variables" in xai.why_this_chart
    assert "user's requested" in xai.why_selected
    assert "Pearson correlation" in xai.what_it_shows
    assert "Correlation does not establish causation." in xai.limitations


def test_faithful_provider_wording_is_accepted_and_receives_no_raw_rows() -> None:
    provider = Provider({
        "why_this_chart": "Scatter plots compare two numeric variables.",
        "what_it_shows": "Across 3 valid pairs, Pearson correlation is 1, a very strong positive relationship.",
        "limitations": ["Correlation does not establish causation.", "The chart is based on 3 valid pairs."],
        "language": "en",
    })
    xai = _scatter(provider)
    assert xai.fallback_used is False
    assert "never-send" not in provider.prompt
    assert "raw_secret" not in provider.prompt


@pytest.mark.parametrize("response", [
    {"why_this_chart": "Scatter plots compare two numeric variables.", "what_it_shows": "The value is 99.", "limitations": [], "language": "en"},
    {"why_this_chart": "Scatter plots compare two numeric variables.", "what_it_shows": "Revenue increased 45%.", "limitations": [], "language": "en"},
    {"why_this_chart": "Scatter plots compare two numeric variables.", "what_it_shows": "Metric Alpha caused Metric Beta.", "limitations": [], "language": "en"},
    {"why_this_chart": "Scatter plots compare two numeric variables.", "what_it_shows": "The relationship is statistically significant.", "limitations": [], "language": "en"},
    {"why_this_chart": "Scatter plots compare two numeric variables.", "what_it_shows": "Mars has the highest value.", "limitations": [], "language": "en"},
])
def test_unverified_provider_claims_fall_back(response) -> None:
    assert _scatter(Provider(response)).fallback_used is True


def test_timeout_unavailable_and_wrong_language_fall_back() -> None:
    assert _scatter(Provider(error=TimeoutError())).fallback_used is True
    wrong = Provider({"why_this_chart": "Gráfico.", "what_it_shows": "Resultado.", "limitations": [], "language": "es"})
    result = _scatter(wrong, language="fr")
    assert result.fallback_used is True and result.language == "en"


def test_automatic_and_requested_results_share_the_additive_xai_contract() -> None:
    frame = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=12, freq="MS"),
        "category": ["A", "B"] * 6,
        "amount": [float(value) for value in range(1, 13)],
        "quantity": [float(value % 4) for value in range(12)],
    })
    automatic = build_visualization_results(frame, dataset_name="generic")
    assert automatic["visualizations"]
    assert all(item["xai"]["visualization_id"] == item["visualization_plan"]["visualization_id"] for item in automatic["visualizations"])

    plan = make_plan("GROUP_AGGREGATE", measures=[AnalyticalExpression(kind="AGGREGATION", operation="SUM", column="amount")], dimension="category")
    requested = build_visualization_results(frame, analytical_plan=plan, dataset_name="generic")
    assert requested["visualizations"][0]["xai"]["status"] == "READY"
    assert "user's requested" in requested["visualizations"][0]["xai"]["why_selected"]
    assert "why_this_chart" in requested["visualizations"][0]["explanation"]
