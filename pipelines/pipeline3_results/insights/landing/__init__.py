"""Dataset-level Insights summary and grounded suggested questions."""

from pipelines.pipeline3_results.insights.landing.schemas import (
    InsightsLandingState,
    NaturalLanguageSummary,
    SuggestedQuestion,
    SuggestedQuestionItem,
    SummaryFact,
    SummaryPackage,
)
from pipelines.pipeline3_results.insights.landing.service import InsightsLandingService
from pipelines.pipeline3_results.insights.landing.suggestions import generate_suggested_questions
from pipelines.pipeline3_results.insights.landing.summary_engine import build_summary_package

__all__ = [
    "InsightsLandingService", "InsightsLandingState", "NaturalLanguageSummary",
    "SuggestedQuestion", "SuggestedQuestionItem", "SummaryFact", "SummaryPackage",
    "build_summary_package", "generate_suggested_questions",
]
