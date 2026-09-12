"""Deterministic post-inference semantic-confidence auditor.

The auditor measures evidentiary support.  It never changes the semantic type
or confidence produced by Architecture B and performs no LLM inference.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT / "benchmarks" / "tabarena" / "pipeline2_artifacts"
    / "low_signal_mic_qwen25_7b.json"
)
DEFAULT_DATASET = ROOT / "benchmarks" / "tabarena" / "datasets" / "MIC.csv"
DEFAULT_OUTPUT = (
    ROOT / "benchmarks" / "tabarena" / "pipeline2_artifacts"
    / "low_signal_mic_semantic_confidence_audit.json"
)
AUDIT_RULE_VERSION = "semantic_confidence_audit_v001"
HIGH_CONFIDENCE = 0.80

DIRECT_LEXICAL_TYPES = {
    "age": {"numeric_measurement"},
    "salary": {"currency", "numeric_measurement"},
    "wage": {"currency", "numeric_measurement"},
    "income": {"currency", "numeric_measurement"},
    "price": {"currency", "numeric_measurement"},
    "cost": {"currency", "numeric_measurement"},
    "amount": {"currency", "numeric_measurement"},
    "temperature": {"numeric_measurement"},
    "temp": {"numeric_measurement"},
    "percentage": {"percentage"},
    "percent": {"percentage"},
    "email": {"contact_information"},
    "phone": {"contact_information"},
    "uuid": {"identifier"},
    "identifier": {"identifier"},
    "date": {"date", "datetime"},
    "datetime": {"datetime"},
    "timestamp": {"datetime"},
    "sex": {"categorical"},
    "gender": {"categorical"},
    "status": {"categorical"},
    "category": {"categorical"},
    "class": {"categorical"},
    "label": {"categorical"},
    "description": {"text"},
}
DOMAIN_ONLY_TOKENS = {"blood", "transaction", "customer", "sensor", "ecg"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def lexical_name_signal(name: str, claimed_type: str) -> dict[str, Any]:
    tokens = [token for token in re.split(r"[^a-z0-9]+", name.casefold()) if token]
    direct = sorted({kind for token in tokens for kind in DIRECT_LEXICAL_TYPES.get(token, set())})
    domain_tokens = sorted(set(tokens) & DOMAIN_ONLY_TOKENS)
    meaningful = bool(direct or domain_tokens)
    if claimed_type in direct:
        strength, relationship = "direct", "supports_claim"
    elif direct:
        strength, relationship = "direct", "supports_other_type"
    elif domain_tokens:
        strength, relationship = "domain_only", "does_not_determine_type"
    else:
        strength, relationship = "none", "opaque_or_coded"
    return {
        "strength": strength,
        "relationship_to_claim": relationship,
        "tokens": tokens,
        "meaningful_name": meaningful,
        "supported_semantic_types": direct,
        "domain_only_tokens": domain_tokens,
        "reason": (
            f"Transparent token(s) directly support {direct}." if direct
            else f"Domain token(s) {domain_tokens} do not determine semantic type." if domain_tokens
            else "No allow-listed transparent lexical token was found; the name is treated as opaque/coded."
        ),
    }


def _numeric_characteristics(series: pd.Series) -> dict[str, Any]:
    non_null = series.dropna()
    numeric = pd.to_numeric(non_null, errors="coerce").dropna()
    conversion_ratio = float(len(numeric) / len(non_null)) if len(non_null) else 0.0
    values = numeric.unique()
    integer_like = bool(len(values)) and all(
        math.isclose(float(value), round(float(value)), abs_tol=1e-9) for value in values
    )
    return {
        "non_null_count": int(len(non_null)),
        "unique_count": int(non_null.nunique(dropna=True)),
        "unique_ratio": float(non_null.nunique(dropna=True) / len(non_null)) if len(non_null) else 0.0,
        "numeric_conversion_ratio": conversion_ratio,
        "integer_like": integer_like,
        "minimum": float(numeric.min()) if len(numeric) else None,
        "maximum": float(numeric.max()) if len(numeric) else None,
    }


def value_shape_signal(series: pd.Series, claimed_type: str) -> dict[str, Any]:
    stats = _numeric_characteristics(series)
    unique = stats["unique_count"]
    numeric = stats["numeric_conversion_ratio"] == 1.0 and stats["non_null_count"] > 0
    plausible: list[str]
    reasons: list[str]
    if numeric and stats["integer_like"] and unique == 2:
        plausible = ["boolean", "categorical", "numeric_measurement", "ordinal"]
        reasons = [
            "Binary numeric values are compatible with boolean flags, category/status codes, ordinal codes, or measured 0/1 counts.",
            "Binary values alone do not prove boolean semantics.",
        ]
        shape_class = "binary_numeric_ambiguous"
    elif numeric and stats["integer_like"] and unique <= 10:
        plausible = ["categorical", "numeric_measurement", "ordinal"]
        reasons = [
            "Low-cardinality integers may encode categories, ordinal levels, or small quantitative counts.",
            "Integer dtype and low cardinality do not uniquely determine semantic type.",
        ]
        shape_class = "low_cardinality_integer_ambiguous"
    elif numeric:
        plausible = ["numeric_measurement", "identifier"]
        if stats["minimum"] is not None and stats["minimum"] >= 0 and stats["maximum"] <= 100:
            plausible.append("percentage")
        reasons = [
            "The values have a numeric shape, but numeric representation alone cannot distinguish measurements, identifiers, or bounded codes."
        ]
        shape_class = "numeric_non_unique"
    else:
        threshold = max(20, int(stats["non_null_count"] * 0.02))
        if unique <= threshold:
            plausible = ["categorical", "text"]
            reasons = ["Repeated string labels are compatible with categorical labels and short text tokens."]
            shape_class = "low_cardinality_string"
        else:
            plausible = ["text", "identifier", "categorical"]
            reasons = ["High-cardinality strings may be free text, identifiers, or sparse categories."]
            shape_class = "high_cardinality_string"
    return {
        "shape_class": shape_class,
        **stats,
        "plausible_semantic_types": plausible,
        "claim_is_shape_compatible": claimed_type in plausible,
        "uniquely_supports_claim": False,
        "reasons": reasons,
    }


def _ratio(values: list[str], predicate: Any) -> float:
    return sum(bool(predicate(value)) for value in values) / len(values) if values else 0.0


def value_pattern_signal(series: pd.Series, claimed_type: str) -> dict[str, Any]:
    values = [str(value).strip() for value in series.dropna().head(2000) if str(value).strip()]
    supported: set[str] = set()
    patterns: list[dict[str, Any]] = []

    checks = [
        ("email", "contact_information", lambda x: re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", x)),
        ("url", "text", lambda x: re.fullmatch(r"https?://\S+", x, re.I)),
        ("uuid", "identifier", lambda x: re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}", x, re.I)),
        ("formatted_percentage", "percentage", lambda x: re.fullmatch(r"[-+]?\d+(?:\.\d+)?\s*%", x)),
        ("formatted_currency", "currency", lambda x: re.fullmatch(r"(?:[$€£₹]\s*\d[\d,]*(?:\.\d+)?)|(?:\d[\d,]*(?:\.\d+)?\s*(?:USD|EUR|GBP|INR))", x, re.I)),
    ]
    for name, kind, predicate in checks:
        ratio = _ratio(values, predicate)
        if ratio >= 0.90:
            supported.add(kind)
            patterns.append({"pattern": name, "match_ratio": ratio, "supports": kind})

    lowered = {value.casefold() for value in values}
    boolean_words = {"true", "false", "yes", "no", "y", "n"}
    if lowered and lowered <= boolean_words and len(lowered) == 2:
        supported.add("boolean")
        patterns.append({"pattern": "explicit_boolean_words", "match_ratio": 1.0, "supports": "boolean"})

    stats = _numeric_characteristics(series)
    nonnumeric = stats["numeric_conversion_ratio"] < 0.5
    threshold = max(20, int(stats["non_null_count"] * 0.02))
    if nonnumeric and 1 < stats["unique_count"] <= threshold:
        supported.add("categorical")
        patterns.append({
            "pattern": "repeated_enumerated_string_labels", "match_ratio": 1.0,
            "supports": "categorical",
        })

    non_null = series.dropna()
    if nonnumeric and len(non_null):
        average_length = sum(len(str(value)) for value in non_null.head(2000)) / min(len(non_null), 2000)
        if average_length >= 30 and stats["unique_ratio"] >= 0.50:
            supported.add("text")
            patterns.append({"pattern": "long_high_cardinality_text", "match_ratio": 1.0, "supports": "text"})

    return {
        "strength": "direct" if supported else "none",
        "supported_semantic_types": sorted(supported),
        "supports_claim": claimed_type in supported,
        "contradicts_claim": bool(supported) and claimed_type not in supported,
        "patterns": patterns,
        "reason": (
            f"Direct deterministic pattern(s) support {sorted(supported)}."
            if supported else "No direct format/content pattern supports a semantic type."
        ),
    }


def rag_or_context_signal(source: dict[str, Any]) -> dict[str, Any]:
    config = source.get("configuration") or {}
    persisted_retrieval = any("retrieved" in call or "rag" in call for call in source.get("calls", []))
    if not config.get("dataset_context_llm_call", True) and not persisted_retrieval:
        return {
            "availability": "unavailable", "strength": "none", "supports_claim": False,
            "reason": "The column-only MIC artifact used no dataset-context call and did not persist retrieved RAG chunks; generic retrieval cannot be counted as corroboration.",
        }
    return {
        "availability": "recorded_but_not_explicitly_linked", "strength": "none",
        "supports_claim": False,
        "reason": "No persisted context statement explicitly links the column to the claimed semantic type.",
    }


def competing_interpretations(
    shape: dict[str, Any], lexical: dict[str, Any], pattern: dict[str, Any]
) -> dict[str, Any]:
    candidates = set(shape["plausible_semantic_types"])
    candidates.update(lexical["supported_semantic_types"])
    candidates.update(pattern["supported_semantic_types"])
    return {
        "count": len(candidates),
        "semantic_types": sorted(candidates),
        "reasons": shape["reasons"] + ([pattern["reason"]] if pattern["patterns"] else []),
    }


def audit_state(
    claimed_type: str, confidence: float, lexical: dict[str, Any],
    shape: dict[str, Any], pattern: dict[str, Any], competing: dict[str, Any],
) -> tuple[str, str]:
    direct_support = (
        lexical["relationship_to_claim"] == "supports_claim" or pattern["supports_claim"]
    )
    if direct_support and not pattern["contradicts_claim"]:
        return "SUPPORTED", "A direct lexical or value-pattern signal independently supports the original semantic type."
    incompatible = not shape["claim_is_shape_compatible"]
    if confidence >= HIGH_CONFIDENCE and (pattern["contradicts_claim"] or incompatible):
        detail = (
            "a direct value pattern supports a different type"
            if pattern["contradicts_claim"] else "the claimed type is not compatible with the observed value shape"
        )
        return "UNSUPPORTED_HIGH_CONFIDENCE", f"Confidence is >= {HIGH_CONFIDENCE:.2f}, but {detail}."
    if competing["count"] > 1:
        return "AMBIGUOUS", "The original type is plausible, but deterministic evidence supports multiple competing interpretations."
    if confidence >= HIGH_CONFIDENCE:
        return "UNSUPPORTED_HIGH_CONFIDENCE", "Confidence is high without an independent lexical, pattern, or uniquely distinguishing shape signal."
    return "AMBIGUOUS", "Available evidence does not uniquely support the original type."


def audit(source: dict[str, Any], dataframe: pd.DataFrame) -> dict[str, Any]:
    records = []
    context_signal = rag_or_context_signal(source)
    for original in source["columns"]:
        name = original["column_name"]
        claimed = original["semantic_type"]
        confidence = float(original["confidence"])
        lexical = lexical_name_signal(name, claimed)
        shape = value_shape_signal(dataframe[name], claimed)
        pattern = value_pattern_signal(dataframe[name], claimed)
        competing = competing_interpretations(shape, lexical, pattern)
        state, reason = audit_state(claimed, confidence, lexical, shape, pattern, competing)
        records.append({
            "column_name": name,
            "original_semantic_type": claimed,
            "original_confidence": confidence,
            "original_reasoning_summary": original.get("rationale"),
            "lexical_name_signal": lexical,
            "value_shape_signal": shape,
            "value_pattern_signal": pattern,
            "rag_or_context_signal": context_signal,
            "competing_interpretations": competing,
            "audit_state": state,
            "deterministic_audit_reason": reason,
            "existing_candidate_heuristic": bool(original.get("fabricated_confidence_candidate")),
        })

    state_counts = Counter(item["audit_state"] for item in records)
    type_breakdown: dict[str, dict[str, int]] = {}
    for item in records:
        bucket = type_breakdown.setdefault(item["original_semantic_type"], {
            "SUPPORTED": 0, "AMBIGUOUS": 0, "UNSUPPORTED_HIGH_CONFIDENCE": 0,
        })
        bucket[item["audit_state"]] += 1
    comparison: dict[str, dict[str, int]] = {
        "existing_candidate_true": {"SUPPORTED": 0, "AMBIGUOUS": 0, "UNSUPPORTED_HIGH_CONFIDENCE": 0},
        "existing_candidate_false": {"SUPPORTED": 0, "AMBIGUOUS": 0, "UNSUPPORTED_HIGH_CONFIDENCE": 0},
    }
    for item in records:
        key = "existing_candidate_true" if item["existing_candidate_heuristic"] else "existing_candidate_false"
        comparison[key][item["audit_state"]] += 1

    return {
        "audit_rule_version": AUDIT_RULE_VERSION,
        "generated_at": utc_now(),
        "source_artifact": source.get("dataset"),
        "source_model": source.get("model"),
        "measurement_only": True,
        "semantic_outputs_modified": False,
        "new_llm_inference": False,
        "thresholds": {"high_confidence": HIGH_CONFIDENCE},
        "summary": {
            "total_columns": len(records),
            "SUPPORTED": state_counts["SUPPORTED"],
            "AMBIGUOUS": state_counts["AMBIGUOUS"],
            "UNSUPPORTED_HIGH_CONFIDENCE": state_counts["UNSUPPORTED_HIGH_CONFIDENCE"],
            "breakdown_by_semantic_type": type_breakdown,
            "existing_candidate_heuristic_count": sum(item["existing_candidate_heuristic"] for item in records),
            "comparison_to_existing_candidate_heuristic": comparison,
        },
        "columns": records,
    }


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "column_name", "original_semantic_type", "original_confidence",
        "lexical_signal", "value_shape_signal", "pattern_signal",
        "rag_or_context_signal", "competing_interpretation_count",
        "competing_interpretations", "audit_state", "deterministic_audit_reason",
        "existing_candidate_heuristic",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in records:
            writer.writerow({
                "column_name": item["column_name"],
                "original_semantic_type": item["original_semantic_type"],
                "original_confidence": item["original_confidence"],
                "lexical_signal": json.dumps(item["lexical_name_signal"], ensure_ascii=False),
                "value_shape_signal": json.dumps(item["value_shape_signal"], ensure_ascii=False),
                "pattern_signal": json.dumps(item["value_pattern_signal"], ensure_ascii=False),
                "rag_or_context_signal": json.dumps(item["rag_or_context_signal"], ensure_ascii=False),
                "competing_interpretation_count": item["competing_interpretations"]["count"],
                "competing_interpretations": json.dumps(item["competing_interpretations"], ensure_ascii=False),
                "audit_state": item["audit_state"],
                "deterministic_audit_reason": item["deterministic_audit_reason"],
                "existing_candidate_heuristic": item["existing_candidate_heuristic"],
            })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    dataframe = pd.read_csv(args.dataset, low_memory=False)
    result = audit(source, dataframe)
    atomic_json(args.output, result)
    write_csv(args.output.with_suffix(".csv"), result["columns"])
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
