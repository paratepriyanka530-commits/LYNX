"""Deterministic whole-column privacy masking."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Sequence

import pandas as pd


MASKING_STRATEGIES = {"hash", "redact"}
REDACTION_VALUE = "[REDACTED]"
HASH_DOMAIN = "lynx-pii-v1"
PII_DEFAULT_STRATEGIES = {
    "email": "hash",
    "phone_number": "hash",
    "national_id": "redact",
    "semantic_contact_information": "hash",
    "semantic_person_name": "redact",
}


def _hash_value(value: Any) -> str:
    payload = f"{HASH_DOMAIN}:{value}".encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def mask_column(
    dataframe: pd.DataFrame,
    target_columns: Sequence[str],
    parameters: Dict[str, Any],
) -> pd.DataFrame:
    """Mask every non-null value in exactly one target column."""
    if len(target_columns) != 1:
        raise ValueError("mask_column requires exactly one target column.")
    if set(parameters) != {"strategy"}:
        raise ValueError("mask_column accepts only the strategy parameter.")
    strategy = parameters.get("strategy")
    if strategy not in MASKING_STRATEGIES:
        raise ValueError("mask_column strategy must be 'hash' or 'redact'.")

    column = target_columns[0]
    source = dataframe[column]
    transform = _hash_value if strategy == "hash" else lambda _: REDACTION_VALUE
    dataframe[column] = source.map(
        lambda value: value if pd.isna(value) else transform(value)
    )
    return dataframe
