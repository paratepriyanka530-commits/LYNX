"""Canonical chart vocabulary shared by production language contracts."""

from typing import Literal


RequestedChart = Literal[
    "BAR", "LINE", "SCATTER", "HISTOGRAM", "COUNT_BAR", "PIE", "DOUGHNUT"
]


__all__ = ["RequestedChart"]
