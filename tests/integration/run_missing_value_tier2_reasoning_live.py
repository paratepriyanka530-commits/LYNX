"""Narrow live diagnostic for the missing-values-only Tier-2 rubric."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import sys
import time
import types
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import requests  # noqa: F401
except ImportError:
    requests_shim = types.ModuleType("requests")

    class RequestException(Exception):
        pass

    class Response:
        def __init__(self, payload, status):
            self._payload = payload
            self.status_code = status

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RequestException(f"HTTP status {self.status_code}")

        def json(self):
            return json.loads(self._payload.decode("utf-8"))

    def post(url, json=None, timeout=None):
        request = urllib.request.Request(
            url,
            data=__import__("json").dumps(json).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return Response(response.read(), response.status)
        except (OSError, urllib.error.URLError) as exc:
            raise RequestException(str(exc)) from exc

    requests_shim.post = post
    requests_shim.RequestException = RequestException
    sys.modules["requests"] = requests_shim

import pandas as pd

from intelligence.llm.ollama_provider import OllamaProvider
import pipelines.pipeline1_cleaning.decisions.cleaning_decision_engine as decision_engine
from pipelines.pipeline1_cleaning.decisions.cleaning_decision_engine import (
    DecisionConfig, _missing_value_readiness, _missing_values_prompt,
    decide_cleaning,
)
from pipelines.pipeline1_cleaning.quality.quality_report import DetectedIssue
from pipelines.pipeline1_cleaning.semantics.batched_schema import SemanticProfile


RESULTS = ROOT / "tests" / "pipeline1_multidomain" / "results"
DATASETS = ROOT / "tests" / "datasets" / "valid"
TARGETS = {
    "dataset_01_retail_transactions": ["total_amount", "customer_id", "transaction_time"],
    "dataset_05_education": ["attendance_percent"],
    "dataset_06_cybersecurity": ["source_port"],
    "dataset_07_iot_sensor": ["error_code"],
    "dataset_08_manufacturing": ["actual_quantity", "defect_quantity", "humidity_percent"],
}
EXPECTED = {
    "total_amount": "action/impute_missing/median",
    "attendance_percent": "action/impute_missing/median",
    "actual_quantity": "action/impute_missing/median",
    "defect_quantity": "action/impute_missing/median",
    "humidity_percent": "action/impute_missing/median",
    "source_port": "insufficient_evidence/no_op",
    "error_code": "flag_for_review/flag_rows",
    "customer_id": "flag_for_review/flag_rows",
    "transaction_time": "flag_for_review/flag_rows",
}


def _memory_snapshot():
    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong), ("memory_load_percent", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong),
            ("available_physical", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong),
            ("available_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("available_extended_virtual", ctypes.c_ulonglong),
        ]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    scale = 1024 ** 3
    return {
        "memory_load_percent": status.memory_load_percent,
        "total_physical_gib": status.total_physical / scale,
        "available_physical_gib": status.available_physical / scale,
        "total_page_file_gib": status.total_page_file / scale,
        "available_page_file_gib": status.available_page_file / scale,
    }


def _loaded_models():
    try:
        with urllib.request.urlopen("http://localhost:11434/api/ps", timeout=5) as response:
            return json.loads(response.read().decode("utf-8")).get("models", [])
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


class RecordingOllamaProvider(OllamaProvider):
    """Capture raw model text without changing production provider behavior."""

    def __init__(self, timeout_seconds=120, record_prompts=False, **kwargs):
        super().__init__(**kwargs)
        self.timeout_seconds = timeout_seconds
        self.record_prompts = record_prompts
        self.call_records = []
        self.last_call_finished_at = None

    def generate_structured(self, prompt, response_model):
        requests_module = sys.modules["requests"]
        original_post = requests_module.post
        call_started = time.perf_counter()
        record = {
            "prompt_character_length": len(prompt), "raw_qwen_response": None,
            "time_since_last_call_seconds": (
                None if self.last_call_finished_at is None
                else call_started - self.last_call_finished_at
            ),
            "pre_call_memory": _memory_snapshot(),
            "pre_call_loaded_models": _loaded_models(),
        }
        if self.record_prompts:
            record["prompt"] = prompt
        self.call_records.append(record)

        def recording_post(*args, **kwargs):
            kwargs["timeout"] = self.timeout_seconds
            payload = kwargs.get("json") or {}
            record["request_keep_alive"] = payload.get("keep_alive")
            record["effective_keep_alive"] = (
                payload.get("keep_alive") if "keep_alive" in payload
                else "Ollama server default (documented default: 5m)"
            )
            response = original_post(*args, **kwargs)
            try:
                record["raw_qwen_response"] = response.json().get("response")
            except Exception:
                pass
            return response

        requests_module.post = recording_post
        try:
            result = super().generate_structured(prompt, response_model)
            record["provider_result"] = result
            return result
        except Exception as exc:
            record["provider_error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.last_call_finished_at = time.perf_counter()
            record["ollama_telemetry"] = dict(self.last_response_metadata or {})
            requests_module.post = original_post


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _numeric_meaning(semantic):
    if semantic["semantic_type"] in {"numeric_measurement", "currency", "percentage"}:
        return "magnitude"
    if semantic["semantic_type"] == "identifier" or semantic["value_function"] == "entity_reference":
        return "identifier"
    return "not_applicable"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--retry-count", type=int, default=1)
    parser.add_argument("--columns", nargs="*")
    parser.add_argument("--record-prompts", action="store_true")
    parser.add_argument("--prompt-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    issues = _jsonl(RESULTS / "issues.jsonl")
    decisions = _jsonl(RESULTS / "decisions.jsonl")
    provider = RecordingOllamaProvider(
        model=args.model, timeout_seconds=args.timeout,
        record_prompts=args.record_prompts,
    )
    output = []

    selected_runs = [
        (repeat_index, dataset_id, columns)
        for repeat_index in range(1, args.repeat + 1)
        for dataset_id, columns in TARGETS.items()
    ]
    for repeat_index, dataset_id, columns in selected_runs:
        columns = [
            column for column in columns
            if not args.columns or column in args.columns
        ]
        if not columns:
            continue
        raw = pd.read_csv(DATASETS / f"{dataset_id}_raw.csv")
        report = json.loads(
            (RESULTS / dataset_id / "cleaning_report.json").read_text(encoding="utf-8")
        )
        semantic_columns = []
        selected_issues = []
        for column in columns:
            issue_data = next(
                item for item in issues
                if item["dataset_id"] == dataset_id
                and item["issue_type"] == "missing_values"
                and item["target_columns"] == [column]
            )
            selected_issues.append(DetectedIssue.model_validate({
                key: value for key, value in issue_data.items()
                if key not in {"dataset_id", "run_id"}
            }))
            decision = next(
                item for item in decisions
                if item["dataset_id"] == dataset_id
                and item["source_issue_ids"] == [issue_data["issue_id"]]
            )
            semantic = dict(decision["semantic_basis"][column])
            semantic["numeric_meaning"] = _numeric_meaning(semantic)
            semantic_columns.append(semantic)

        profile = SemanticProfile.model_validate({
            "dataset_context": report["semantic_profile"]["dataset_context"],
            "columns": semantic_columns,
        })
        if args.prompt_only:
            entry = {
                "dataset_id": dataset_id, "repeat": repeat_index,
                "columns": columns,
                "prompt": _missing_values_prompt(selected_issues, profile, raw),
            }
            output.append(entry)
            print(entry["prompt"], flush=True)
            continue
        call_start = len(provider.call_records)
        entry = {
            "dataset_id": dataset_id, "repeat": repeat_index,
            "model": args.model, "calls": [], "cases": [],
        }
        readiness = {
            issue.issue_id: _missing_value_readiness(issue, profile, raw)
            for issue in selected_issues
        }
        # The earlier nine-case diagnostic deliberately sent every selected case
        # through Tier 2, including cases normally resolved by Tier 1. Preserve
        # that diagnostic boundary while exercising the production recovery loop.
        original_tier1 = decision_engine._tier1
        decision_engine._tier1 = lambda issue, semantic: None
        try:
            final_decisions, summary, warnings = decide_cleaning(
                selected_issues, profile, provider,
                config=DecisionConfig(retry_count=args.retry_count), dataframe=raw,
            )
        finally:
            decision_engine._tier1 = original_tier1

        audit = summary["tier2_recovery_audit"][0]
        provider_calls = provider.call_records[call_start:]
        for call_audit, provider_call in zip(audit["calls"], provider_calls):
            entry["calls"].append({**call_audit, **provider_call})

        final_index = {
            decision.source_issue_ids[0]: decision for decision in final_decisions
        }
        for issue in selected_issues:
            issue_calls = [
                call for call in entry["calls"] if issue.issue_id in call["issue_ids"]
            ]
            first = issue_calls[0]
            first_rejection = next((
                item["reason"] for item in first.get("rejected", [])
                if item["issue_id"] == issue.issue_id
            ), None)
            retry_calls = issue_calls[1:]
            final = final_index[issue.issue_id]
            entry["cases"].append({
                "column": issue.target_columns[0],
                "expected": EXPECTED[issue.target_columns[0]],
                "readiness": readiness[issue.issue_id],
                "first_raw_qwen_response": first.get("raw_qwen_response"),
                "first_validation": (
                    "accepted" if issue.issue_id in first.get("accepted_issue_ids", [])
                    else first_rejection or first.get("provider_error")
                ),
                "accepted_immediately": issue.issue_id in audit["accepted_on_first_response"],
                "retried": bool(retry_calls),
                "retry_attempts": [{
                    "stage": call["stage"],
                    "retry_batch_size": call["retry_batch_size"],
                    "raw_qwen_response": call.get("raw_qwen_response"),
                    "validation": (
                        "accepted" if issue.issue_id in call.get("accepted_issue_ids", [])
                        else next((
                            item["reason"] for item in call.get("rejected", [])
                            if item["issue_id"] == issue.issue_id
                        ), call.get("provider_error"))
                    ),
                } for call in retry_calls],
                "final_decision": final.model_dump(mode="json"),
                "fallback": final.decision_source == "fallback",
            })
        entry["summary"] = summary
        entry["warnings"] = warnings
        output.append(entry)
        print(json.dumps(entry, ensure_ascii=False, indent=2, default=str), flush=True)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(output, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    print("COMPACT_RESULTS")
    for entry in output:
        if "prompt" in entry:
            continue
        if "error" in entry:
            print(f"{entry['dataset_id']}: ERROR {entry['error']}")
            continue
        for case in entry["cases"]:
            decision = case["final_decision"]
            readiness = case["readiness"]
            print(
                f"{case['column']}: {decision['decision_type']} / "
                f"{decision['proposed_action']} / {decision['parameters']} | "
                f"semantic={readiness['median_semantic_eligibility']} | "
                f"mechanical={readiness['median_mechanical_preflight']} | "
                f"rationale={decision['rationale']}"
            )


if __name__ == "__main__":
    main()
