"""Production prompt for explaining deterministic analytical results."""

from __future__ import annotations

import json

from pipelines.pipeline3_results.insights.response_generation.schemas import (
    ResponseContext,
)


SYSTEM_PROMPT = """You are LYNX's result explanation layer.

You receive a verified deterministic analytical result.

Rules:
1. Answer only from the supplied verified result.
2. Never recalculate.
3. Never modify numeric values.
4. Never invent missing information.
5. Never infer causality unless explicitly provided as verified evidence.
6. Do not claim a trend, ranking, percentage, or comparison that is absent from the result.
7. Mention important caveats when supplied.
8. Use natural user-friendly language.
9. Respond in the requested/user language when possible.
10. Keep answers concise unless the user asks for detail.
11. If the result is empty or partial, say so clearly.
12. Do not expose internal implementation details or chain-of-thought.

Return one JSON object with exactly two fields: text and language. Copy all numbers,
percentages, and entity names exactly as supplied. Do not output markdown or analysis."""


def build_response_prompt(context: ResponseContext) -> str:
    payload = context.model_dump(mode="json")
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "Verified response context:\n"
        f"{json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}"
    )


__all__ = ["SYSTEM_PROMPT", "build_response_prompt"]
