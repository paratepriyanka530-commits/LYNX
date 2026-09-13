"""Validate model artifact/runtime compatibility declarations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_MODELS = {
    "random_forest",
    "lightgbm",
    "xgboost",
    "catboost",
    "tabpfn",
    "xtab",
    "realmlp",
    "tabm",
    "tabiclv2",
    "ft_transformer",
}
DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "model_requirements_manifest.json"


def check_manifest(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("Manifest field 'candidates' must be a list.")

    ids = [entry.get("model_id") for entry in candidates]
    failures = []
    if len(ids) != len(set(ids)):
        failures.append("duplicate model_id values are present")
    missing = sorted(EXPECTED_MODELS - set(ids))
    unexpected = sorted(set(ids) - EXPECTED_MODELS)
    if missing:
        failures.append(f"missing candidates: {missing}")
    if unexpected:
        failures.append(f"unexpected candidates: {unexpected}")
    if payload.get("candidate_count") != len(candidates):
        failures.append(
            "candidate_count does not equal the number of candidate records"
        )

    for entry in candidates:
        model_id = entry.get("model_id", "<missing-model-id>")
        matches = entry.get("artifact_environment_matches_runtime")
        override = entry.get("verified_safe_despite_mismatch")
        mismatches = entry.get("mismatches")
        evidence = entry.get("verification_evidence")

        if not isinstance(matches, bool):
            failures.append(
                f"{model_id}: artifact_environment_matches_runtime must be boolean"
            )
            continue
        if not isinstance(override, bool):
            failures.append(
                f"{model_id}: verified_safe_despite_mismatch must be boolean"
            )
            continue
        if not isinstance(mismatches, list):
            failures.append(f"{model_id}: mismatches must be a list")
            continue

        if matches:
            print(f"PASS {model_id}: artifact environment matches runtime")
            continue
        if override and mismatches and evidence:
            print(
                f"PASS {model_id}: mismatch explicitly verified safe - "
                + "; ".join(mismatches)
            )
            continue
        failures.append(
            f"{model_id}: artifact/runtime mismatch is not explicitly verified "
            f"safe ({'; '.join(mismatches) if mismatches else 'details missing'})"
        )

    if failures:
        print("MODEL REQUIREMENTS MANIFEST CHECK FAILED")
        for failure in failures:
            print(f"FAIL {failure}")
        return 1

    print(f"MODEL REQUIREMENTS MANIFEST CHECK PASSED: {len(candidates)} models")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    args = parser.parse_args()
    return check_manifest(args.manifest.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
