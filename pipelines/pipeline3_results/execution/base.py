"""Execution-engine-neutral interface for analytical plans."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from pipelines.pipeline3_results.execution.schemas import VerifiedResult
from pipelines.pipeline3_results.grounding.schemas import DatasetKnowledge
from pipelines.pipeline3_results.planning.schemas import AnalyticalPlan


class AnalyticalExecutor(ABC):
    @abstractmethod
    def execute(
        self,
        plan: AnalyticalPlan,
        dataframe: pd.DataFrame,
        dataset_knowledge: DatasetKnowledge,
    ) -> VerifiedResult:
        raise NotImplementedError


__all__ = ["AnalyticalExecutor"]
