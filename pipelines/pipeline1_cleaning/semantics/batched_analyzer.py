"""Bounded batched semantic analysis for Pipeline 1 Architecture B."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from intelligence.llm.provider import LLMProvider
from intelligence.rag.retriever import RAGRetriever
from pipelines.pipeline1_cleaning.semantics.batched_schema import (
    BatchedColumnInference, BatchedColumnSemantic, BatchedDatasetContext, SemanticProfile,
)
from pipelines.pipeline1_cleaning.semantics.dataset_understanding import (
    build_dataset_semantic_prompt, retrieve_dataset_context,
)
from pipelines.pipeline1_cleaning.semantics.semantic_schema import DatasetSemanticInference


@dataclass(frozen=True)
class BatchSemanticConfig:
    max_columns_per_batch: int = 12
    sample_cap: int = 5
    batch_rag_top_k: int = 8
    dataset_rag_top_k: int = 8
    retry_count: int = 1


DEFAULT_CONFIG = BatchSemanticConfig()


class BatchCoverageError(ValueError):
    pass


def build_compact_column_evidence(dataset_profile: Dict[str, Any], sample_cap: int = 5) -> List[Dict[str, Any]]:
    compact = []
    for column in dataset_profile["column_profiles"]:
        item = {
            "column_name": column["column_name"],
            "physical_dtype": column["physical_dtype"],
            "missing_percentage": column["missing"]["percentage"],
            "unique_ratio": column["uniqueness"]["unique_ratio"],
            "repetition_rate": column["repetition"]["repetition_rate"],
            "sample_values": column["sample_values"][:sample_cap],
        }
        if "numeric_statistics" in column:
            item["numeric_summary"] = column["numeric_statistics"]
        compact.append(item)
    return compact


def create_batches(columns: Sequence[Dict[str, Any]], max_columns_per_batch: int) -> List[List[Dict[str, Any]]]:
    if max_columns_per_batch <= 0:
        raise ValueError("max_columns_per_batch must be greater than zero.")
    return [list(columns[i:i + max_columns_per_batch]) for i in range(0, len(columns), max_columns_per_batch)]


def validate_batch_coverage(expected_names: Sequence[str], results: Sequence[BatchedColumnSemantic]) -> Dict[str, BatchedColumnSemantic]:
    actual = [result.column_name for result in results]
    duplicates = sorted({name for name in actual if actual.count(name) > 1})
    missing = sorted(set(expected_names) - set(actual))
    invented = sorted(set(actual) - set(expected_names))
    if duplicates or missing or invented:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if duplicates:
            details.append("duplicate=" + ",".join(duplicates))
        if invented:
            details.append("invented=" + ",".join(invented))
        raise BatchCoverageError("Invalid semantic batch coverage: " + "; ".join(details))
    indexed = {result.column_name: result for result in results}
    return {name: indexed[name] for name in expected_names}


def build_batch_rag_query(columns: Sequence[Dict[str, Any]], context: BatchedDatasetContext) -> str:
    return (
        "Retrieve semantic definitions useful for these tabular columns. Distinguish entity references, "
        "classifications, measurements, temporal values, descriptions, contacts and locations.\n"
        f"Row entity: {context.row_entity}\nProbable domain: {context.probable_domain}\n"
        f"Column evidence: {list(columns)}"
    )


def build_batch_prompt(columns: Sequence[Dict[str, Any]], context: BatchedDatasetContext, rag: Sequence[Dict[str, Any]]) -> str:
    knowledge = "\n\n".join(f"[{x['topic']} | {x['section']}]\n{x['content']}" for x in rag)
    return f"""
You are the batched column semantic analyzer for an unknown tabular dataset.
Return exactly one result for every input column using its exact column_name.
Do not omit, duplicate, rename, or invent columns. Match results by name.

For each column first infer what its values represent, then assign the matching
value_function, semantic_type, and numeric_meaning. concept_description must
briefly describe the values' meaning and reasoning_summary must cite decisive
evidence, including whether arithmetic and central tendency are meaningful for
numeric-looking values.

semantic_role is not being resolved in this analyzer. Return
semantic_role="unknown" for every column and spend no reasoning on structural
roles such as primary or foreign identifier.

Choose one value_function by meaning:
- entity_reference: values identify particular entities, records, or objects;
- classification: reusable classes, labels, states, types, levels, or groups;
- measurement: observed or derived quantitative magnitude;
- temporal: dates, times, datetimes, or temporal position;
- descriptive_attribute: descriptive or narrative content;
- boolean_state: a binary logical state;
- contact: contact information;
- location: geographic or physical location;
- unknown: evidence does not support another function.

Choose one semantic_type by meaning:
- identifier: values identify or reference particular instances;
- numeric_measurement: numeric quantitative magnitude;
- categorical: reusable categories, codes, labels, types, states, or levels;
- text: descriptive or narrative text;
- date, time, datetime: their corresponding temporal representation;
- boolean: binary logical values;
- currency: monetary amounts;
- percentage: proportions, rates, ratios, or explicit percentages;
- contact_information: email, phone, or other contact values;
- location: geographic or physical locations;
- unknown: evidence genuinely supports none of the defined types.

Choose one numeric_meaning by meaning:
- magnitude: arithmetic differences and central tendency represent a meaningful
  quantitative amount, rate, or measurement;
- ordinal: order is meaningful but numeric distance is not necessarily meaningful;
- nominal_code: numeric representation selects categories, states, addresses, or
  labels, so arithmetic is meaningless;
- identifier: numeric representation identifies an instance, entity, or record;
- not_applicable: values are not semantically numeric;
- unknown: evidence is insufficient to determine how numeric values function.

Numeric storage or numeric-looking values do not by themselves establish
numeric_meaning="magnitude". Decide whether arithmetic describes the represented
concept. Keep numeric_meaning consistent with semantic_type, value_function,
concept_description, and reasoning_summary.

Identifier-shaped values may instead classify records, and numeric dtype may
instead encode identifiers or categories. value_function describes meaning,
not primary/foreign structural role.

Keep concept_description and value_function consistent. If the evidence supports
a particular entity identifier, use entity_reference; a reusable type, label,
size, state, or category uses classification; a date or time uses temporal; a
quantity or amount uses measurement; descriptive names or contents use
descriptive_attribute. Use unknown only when the value meaning itself cannot be
assigned to any defined function.

Observed evidence is authoritative. Column name is evidence but not proof.
Physical dtype alone does not determine meaning. A numeric column can be an
identifier, a string can represent a date, and a code can be a classification
rather than an entity reference. Use unknown only when evidence genuinely does
not support a reasonable interpretation; it is not the default choice. Do not
detect issues, recommend cleaning, or transform data.

SHARED DATASET CONTEXT
Row entity: {context.row_entity}
Row-entity confidence: {context.row_entity_confidence}
Probable domain (soft only): {context.probable_domain}
Domain confidence: {context.domain_confidence}
Dataset description: {context.dataset_description}

OBSERVED COLUMN EVIDENCE
{list(columns)}

RAG KNOWLEDGE — general guidance, not dataset evidence
{knowledge}

Return only the required structured fields.
""".strip()


def _unknown(name: str) -> BatchedColumnSemantic:
    return BatchedColumnSemantic(
        column_name=name, semantic_type="unknown", value_function="unknown",
        numeric_meaning="unknown",
        concept_description="Semantic analysis unavailable for this column.",
        semantic_role="unknown", confidence=0.0,
        reasoning_summary="The semantic batch failed validation after retry.",
    )


def _metrics(provider: LLMProvider, wall: float) -> Dict[str, Any]:
    raw = getattr(provider, "last_response_metadata", None) or {}
    result = {"api_wall_duration_seconds": round(wall, 6)}
    for key in ("prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration", "total_duration", "load_duration", "done_reason"):
        result[key] = raw.get(key)
    return result


def analyze_batched_semantics(
    dataset_profile: Dict[str, Any], dataset_evidence: Dict[str, Any],
    retriever: RAGRetriever, llm_provider: LLMProvider,
    config: BatchSemanticConfig = DEFAULT_CONFIG,
) -> SemanticProfile:
    started = time.perf_counter()
    warnings: List[str] = []
    telemetry: Dict[str, Any] = {
        "dataset_llm_calls": 0, "column_batch_llm_calls": 0, "total_llm_calls": 0,
        "batch_sizes": [], "dataset_semantic_runtime_seconds": None,
        "batch_runtime_seconds": [], "rag_runtime_seconds": 0.0,
        "batch_failures": 0, "batch_retries": 0, "llm_calls": [],
    }

    dataset_started = time.perf_counter()
    try:
        rag_started = time.perf_counter()
        dataset_rag = retrieve_dataset_context(dataset_evidence, retriever, config.dataset_rag_top_k)
        telemetry["rag_runtime_seconds"] += time.perf_counter() - rag_started
        prompt = build_dataset_semantic_prompt(dataset_evidence, dataset_rag)
        call_started = time.perf_counter()
        data = llm_provider.generate_structured(prompt, DatasetSemanticInference)
        wall = time.perf_counter() - call_started
        telemetry["dataset_llm_calls"] += 1
        telemetry["llm_calls"].append({"stage": "dataset_context", **_metrics(llm_provider, wall)})
        domain, confidence = data.get("dataset_domain"), data["confidence"]
        context = BatchedDatasetContext(
            row_entity=data.get("row_entity"), row_entity_confidence=confidence,
            probable_domain=domain, domain_confidence=confidence if domain else None,
            dataset_description=data.get("dataset_description"),
        )
    except Exception as exc:
        telemetry["dataset_llm_calls"] += 1
        warnings.append(f"Dataset semantic context failed: {type(exc).__name__}: {exc}")
        context = BatchedDatasetContext(row_entity=None, row_entity_confidence=0.0)
    telemetry["dataset_semantic_runtime_seconds"] = round(time.perf_counter() - dataset_started, 6)

    compact = build_compact_column_evidence(dataset_profile, config.sample_cap)
    results: List[BatchedColumnSemantic] = []
    for batch_number, batch in enumerate(create_batches(compact, config.max_columns_per_batch), 1):
        expected = [item["column_name"] for item in batch]
        telemetry["batch_sizes"].append(len(batch))
        rag_started = time.perf_counter()
        retrieved = retriever.retrieve(
            build_batch_rag_query(batch, context), top_k=config.batch_rag_top_k,
            categories=["semantic", "semantic_concept"],
        )
        telemetry["rag_runtime_seconds"] += time.perf_counter() - rag_started
        deduplicated, seen = [], set()
        for item in retrieved:
            if item.get("chunk_id") not in seen:
                seen.add(item.get("chunk_id"))
                deduplicated.append(item)
        prompt = build_batch_prompt(batch, context, deduplicated)
        batch_started, resolved, last_error = time.perf_counter(), None, None
        for attempt in range(config.retry_count + 1):
            if attempt:
                telemetry["batch_retries"] += 1
            telemetry["column_batch_llm_calls"] += 1
            call_started = time.perf_counter()
            try:
                raw = llm_provider.generate_structured(prompt, BatchedColumnInference)
                wall = time.perf_counter() - call_started
                telemetry["llm_calls"].append({"stage": "column_batch", "batch_index": batch_number, "attempt": attempt + 1, **_metrics(llm_provider, wall)})
                parsed = BatchedColumnInference(**raw)
                resolved = validate_batch_coverage(expected, parsed.columns)
                break
            except Exception as exc:
                wall = time.perf_counter() - call_started
                telemetry["llm_calls"].append({"stage": "column_batch", "batch_index": batch_number, "attempt": attempt + 1, "error": f"{type(exc).__name__}: {exc}", **_metrics(llm_provider, wall)})
                last_error = exc
        if resolved is None:
            telemetry["batch_failures"] += 1
            warnings.append(f"Semantic batch {batch_number} failed after retry; unknown fallback used for {expected}: {type(last_error).__name__}: {last_error}")
            results.extend(_unknown(name) for name in expected)
        else:
            results.extend(resolved[name] for name in expected)
        telemetry["batch_runtime_seconds"].append(round(time.perf_counter() - batch_started, 6))

    telemetry["total_llm_calls"] = telemetry["dataset_llm_calls"] + telemetry["column_batch_llm_calls"]
    telemetry["rag_runtime_seconds"] = round(telemetry["rag_runtime_seconds"], 6)
    telemetry["total_semantic_runtime_seconds"] = round(time.perf_counter() - started, 6)
    return SemanticProfile(dataset_context=context, columns=results, warnings=warnings, telemetry=telemetry)
