"""Secure run persistence and Pipeline 1 execution for the LYNX application."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
from threading import RLock
from typing import Any, Callable, Mapping
from uuid import UUID, uuid4
import zipfile

import pandas as pd
from starlette.datastructures import UploadFile

from app.config import ApplicationSettings
from pipelines.pipeline1_cleaning.ingestion.file_validator import (
    ALLOWED_EXTENSIONS,
    validate_file,
)


Pipeline1Runner = Callable[[str | Path, str | Path], Any]
ACTIVE_STATES = {"queued", "running"}
FINAL_STATES = {"complete", "failed"}
_SAFE_DISPLAY_CHARS = re.compile(r"[^\w.()\[\] -]+", re.UNICODE)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_pipeline1_runner(input_path: str | Path, output_directory: str | Path) -> Any:
    """Import the heavy semantic stack only when a Pipeline 1 worker needs it."""
    from pipelines.pipeline1_cleaning.pipeline import run_pipeline1

    return run_pipeline1(input_path, output_directory)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        return dumped if isinstance(dumped, Mapping) else {}
    return {}


class ServiceError(RuntimeError):
    """Controlled application error safe to return through the HTTP boundary."""

    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class RunStore:
    """Filesystem-backed state store with atomic JSON writes and containment checks."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks_guard = RLock()
        self._run_locks: dict[str, RLock] = {}

    def _run_lock(self, run_id: str) -> RLock:
        """Return the stable in-process lock for one normalized run ID."""
        normalized = self.validate_run_id(run_id)
        with self._locks_guard:
            return self._run_locks.setdefault(normalized, RLock())

    @contextmanager
    def locked_run(self, run_id: str):
        """Hold one run's re-entrant lock across a compound service operation."""
        normalized = self.validate_run_id(run_id)
        with self._run_lock(normalized):
            yield normalized

    def validate_run_id(self, run_id: str) -> str:
        try:
            parsed = UUID(str(run_id))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ServiceError(400, "malformed_run_id", "Run ID must be a valid UUID.") from exc
        return str(parsed)

    def _contained(self, path: Path) -> Path:
        resolved = path.resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ServiceError(
                400, "invalid_run_path", "Resolved run path is outside the runtime root."
            ) from exc
        return resolved

    def run_directory(self, run_id: str, *, require_exists: bool = True) -> Path:
        normalized = self.validate_run_id(run_id)
        path = self._contained(self.root / normalized)
        if require_exists and not path.is_dir():
            raise ServiceError(404, "run_not_found", "The requested run does not exist.")
        return path

    def paths(self, run_id: str) -> dict[str, Path]:
        run_dir = self.run_directory(run_id)
        state = self._contained(run_dir / "state.json")
        upload_dir = self._contained(run_dir / "upload")
        pipeline1_dir = self._contained(run_dir / "pipeline1")
        return {
            "run": run_dir,
            "state": state,
            "upload": upload_dir,
            "pipeline1": pipeline1_dir,
        }

    def _atomic_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        destination = self._contained(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._contained(
            destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        )
        try:
            serialized = json.dumps(payload, indent=2, ensure_ascii=False)
            with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            # The temporary file is closed before replacement. Both paths are
            # in the same directory, preserving atomic replacement semantics.
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def create(self, original_filename: str, extension: str) -> dict[str, Any]:
        run_id = str(uuid4())
        with self._run_lock(run_id):
            run_dir = self.run_directory(run_id, require_exists=False)
            run_dir.mkdir(parents=False, exist_ok=False)
            (run_dir / "upload").mkdir()
            (run_dir / "pipeline1").mkdir()
            now = _utc_now()
            state = {
                "application_run_id": run_id,
                "state": "queued",
                "stage": "upload_validation",
                "message": "Dataset accepted and awaiting Pipeline 1 execution.",
                "created_at": now,
                "updated_at": now,
                "original_filename": original_filename,
                "stored_extension": extension,
                "pipeline1_internal_run_id": None,
                "error": None,
            }
            self._atomic_json(run_dir / "state.json", state)
            return state

    def _read_unlocked(self, run_id: str) -> dict[str, Any]:
        path = self.paths(run_id)["state"]
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ServiceError(404, "run_not_found", "The requested run does not exist.") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ServiceError(500, "run_state_unavailable", "Run state is unavailable.") from exc
        return payload

    def read(self, run_id: str) -> dict[str, Any]:
        with self._run_lock(run_id):
            return self._read_unlocked(run_id)

    def update(self, run_id: str, **changes: Any) -> dict[str, Any]:
        with self._run_lock(run_id):
            state = self._read_unlocked(run_id)
            state.update(changes)
            state["updated_at"] = _utc_now()
            self._atomic_json(self.paths(run_id)["state"], state)
            return state

    def remove(self, run_id: str) -> bool:
        with self._run_lock(run_id):
            try:
                run_dir = self.run_directory(run_id)
            except ServiceError as exc:
                if exc.code == "run_not_found":
                    return False
                raise
            state = self._read_unlocked(run_id)
            if state.get("state") in ACTIVE_STATES:
                return False
            # Resolve and recheck immediately before recursive removal.
            target = self._contained(run_dir)
            if target == self.root:
                raise ServiceError(400, "invalid_run_path", "Runtime root cannot be removed.")
            shutil.rmtree(target)
            return True

    def discard_incomplete_creation(self, run_id: str) -> None:
        """Remove a rejected upload run, including a partial temporary upload."""
        with self._run_lock(run_id):
            try:
                run_dir = self.run_directory(run_id)
            except ServiceError as exc:
                if exc.code == "run_not_found":
                    return
                raise
            target = self._contained(run_dir)
            if target != self.root:
                shutil.rmtree(target)

    def cleanup_expired(
        self, retention_seconds: int, *, now: datetime | None = None,
    ) -> list[str]:
        if retention_seconds <= 0:
            raise ValueError("retention_seconds must be greater than zero")
        reference = now or datetime.now(timezone.utc)
        removed: list[str] = []
        for child in list(self.root.iterdir()):
            if not child.is_dir():
                continue
            try:
                run_id = self.validate_run_id(child.name)
                state = self.read(run_id)
                if state.get("state") in ACTIVE_STATES:
                    continue
                updated = datetime.fromisoformat(str(state["updated_at"]))
                if updated.tzinfo is None:
                    updated = updated.replace(tzinfo=timezone.utc)
                if (reference - updated).total_seconds() < retention_seconds:
                    continue
                if self.remove(run_id):
                    removed.append(run_id)
            except (ServiceError, ValueError, KeyError, OSError):
                # Unknown/non-run directories and corrupt state are never deleted.
                continue
        return removed


def sanitize_display_filename(filename: str | None, extension: str) -> str:
    raw = str(filename or "")
    basename = PurePosixPath(raw.replace("\\", "/")).name
    cleaned = "".join(character for character in basename if character.isprintable())
    cleaned = _SAFE_DISPLAY_CHARS.sub("_", cleaned).strip(" .")
    if not cleaned:
        cleaned = f"dataset{extension}"
    if not cleaned.casefold().endswith(extension.casefold()):
        cleaned = f"{cleaned[:200]}{extension}"
    return cleaned[:255]


def _is_absolute_local_path(value: str) -> bool:
    return PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute()


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _sanitize_value(item)
            for key, item in value.items()
            if key not in {"source_path", "absolute_path"}
        }
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, str) and _is_absolute_local_path(value):
        return "[local path removed]"
    return value


def sanitize_pipeline1_report(
    report: Mapping[str, Any], run_id: str,
) -> dict[str, Any]:
    """Create a presentation copy without mutating the canonical report."""
    sanitized = _sanitize_value(json.loads(json.dumps(report, default=str)))
    sanitized["output_artifacts"] = {
        "cleaned_dataset": None,
        "cleaning_report": f"/api/runs/{run_id}/pipeline1",
        "cleaned_dataset_format": "csv",
    }
    return sanitized


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_eligibility(
    report: Mapping[str, Any], cleaned_path: Path, *, columns_unique: bool,
) -> dict[str, Any]:
    reasons: list[str] = []
    if report.get("status") != "success":
        reasons.append("pipeline1_status_not_success")
    if not bool(_as_mapping(report.get("validation")).get("passed")):
        reasons.append("post_clean_validation_not_passed")
    privacy = _as_mapping(
        _as_mapping(report.get("decision_summary")).get("privacy_actions")
    )
    if not bool(privacy.get("privacy_stage_completed")):
        reasons.append("privacy_stage_not_completed")
    if not bool(privacy.get("privacy_clearance_passed")):
        reasons.append("privacy_clearance_not_passed")
    if not columns_unique:
        reasons.append("prepared_dataset_has_duplicate_columns")

    expected_digest = privacy.get("cleaned_dataset_sha256")
    digest_verified = False
    if not cleaned_path.is_file():
        reasons.append("cleaned_dataset_missing")
    elif not isinstance(expected_digest, str) or not expected_digest:
        reasons.append("cleaned_dataset_hash_missing")
    else:
        digest_verified = _sha256(cleaned_path) == expected_digest
        if not digest_verified:
            reasons.append("cleaned_dataset_hash_mismatch")
    return {
        "eligible": not reasons,
        "reasons": reasons,
        "cleaned_dataset_sha256_verified": digest_verified,
    }


class ApplicationRunService:
    """Own upload persistence, bounded Pipeline 1 work, and result retrieval."""

    def __init__(
        self,
        settings: ApplicationSettings | None = None,
        *,
        pipeline1_runner: Pipeline1Runner = _default_pipeline1_runner,
    ):
        self.settings = settings or ApplicationSettings.from_environment()
        self.store = RunStore(self.settings.runtime_root)
        self.pipeline1_runner = pipeline1_runner
        self.executor = ThreadPoolExecutor(
            max_workers=self.settings.pipeline1_worker_count,
            thread_name_prefix="lynx-pipeline1",
        )

    @staticmethod
    def _extension(filename: str | None) -> str:
        # The basename is used only to inspect the suffix, never as a path.
        basename = PurePosixPath(str(filename or "").replace("\\", "/")).name
        return Path(basename).suffix.casefold()

    def _xlsx_safety_check(self, path: Path) -> None:
        if not zipfile.is_zipfile(path):
            raise ServiceError(422, "invalid_spreadsheet", "XLSX file is not a valid workbook archive.")
        try:
            with zipfile.ZipFile(path) as workbook:
                entries = workbook.infolist()
                if len(entries) > self.settings.xlsx_max_entries:
                    raise ServiceError(413, "xlsx_expansion_limit", "XLSX workbook contains too many archive entries.")
                uncompressed = sum(entry.file_size for entry in entries)
                compressed = sum(entry.compress_size for entry in entries)
                if uncompressed > self.settings.xlsx_max_uncompressed_bytes:
                    raise ServiceError(413, "xlsx_expansion_limit", "XLSX expanded size exceeds the configured limit.")
                ratio = uncompressed / max(1, compressed)
                if ratio > self.settings.xlsx_max_compression_ratio:
                    raise ServiceError(413, "xlsx_compression_ratio", "XLSX compression ratio exceeds the configured limit.")
        except zipfile.BadZipFile as exc:
            raise ServiceError(422, "invalid_spreadsheet", "XLSX workbook archive is malformed.") from exc

    async def create_from_upload(self, upload: UploadFile) -> dict[str, Any]:
        extension = self._extension(upload.filename)
        if extension not in ALLOWED_EXTENSIONS:
            await upload.close()
            raise ServiceError(
                415, "unsupported_file_type",
                "Only .csv, .xlsx, and .xls dataset files are supported.",
            )
        display_name = sanitize_display_filename(upload.filename, extension)
        state = self.store.create(display_name, extension)
        run_id = state["application_run_id"]
        paths = self.store.paths(run_id)
        final_path = paths["upload"] / f"dataset{extension}"
        temporary = paths["upload"] / f".upload-{uuid4().hex}.tmp"
        total = 0
        try:
            with temporary.open("xb") as destination:
                while True:
                    chunk = await upload.read(self.settings.upload_chunk_bytes)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > self.settings.maximum_upload_bytes:
                        raise ServiceError(
                            413, "upload_too_large", "Dataset exceeds the configured upload limit."
                        )
                    destination.write(chunk)
            if total == 0:
                raise ServiceError(422, "empty_upload", "Dataset upload is empty.")
            os.replace(temporary, final_path)
            if extension == ".xlsx":
                self._xlsx_safety_check(final_path)
            validation = await asyncio.to_thread(validate_file, str(final_path))
            if not validation.get("valid"):
                raise ServiceError(
                    422, "invalid_dataset",
                    "Dataset could not be parsed or did not satisfy Pipeline 1 input requirements.",
                )
        except ServiceError:
            temporary.unlink(missing_ok=True)
            self.store.discard_incomplete_creation(run_id)
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            temporary.unlink(missing_ok=True)
            self.store.discard_incomplete_creation(run_id)
            raise ServiceError(
                422, "upload_processing_failed", "Dataset upload could not be processed safely."
            ) from exc
        finally:
            await upload.close()

        self.executor.submit(self._execute_pipeline1, run_id)
        return {
            "run_id": run_id,
            "state": "queued",
            "status_url": f"/api/runs/{run_id}",
        }

    def _execute_pipeline1(self, run_id: str) -> None:
        paths = self.store.paths(run_id)
        state = self.store.read(run_id)
        input_path = paths["upload"] / f"dataset{state['stored_extension']}"
        self.store.update(
            run_id,
            state="running",
            stage="pipeline1",
            message="Pipeline 1 is analyzing and preparing the dataset.",
            error=None,
        )
        try:
            returned_report = self.pipeline1_runner(input_path, paths["pipeline1"])
            report_path = paths["pipeline1"] / "cleaning_report.json"
            cleaned_path = paths["pipeline1"] / "cleaned_dataset.csv"
            if report_path.is_file():
                report = json.loads(report_path.read_text(encoding="utf-8"))
            else:
                report = dict(_as_mapping(returned_report))
            internal_run_id = report.get("run_id")
            if not internal_run_id or not report_path.is_file() or not cleaned_path.is_file():
                raise RuntimeError("Pipeline 1 did not produce its required artifacts")
            if report.get("status") != "success":
                self.store.update(
                    run_id,
                    state="failed",
                    stage="pipeline1",
                    message="Pipeline 1 completed with a failed validation status.",
                    pipeline1_internal_run_id=str(internal_run_id),
                    error={
                        "code": "pipeline1_validation_failed",
                        "message": "Pipeline 1 could not produce a validated prepared dataset.",
                    },
                )
                return
            self.store.update(
                run_id,
                state="complete",
                stage="pipeline1_complete",
                message="Pipeline 1 completed successfully.",
                pipeline1_internal_run_id=str(internal_run_id),
                error=None,
            )
        except Exception:
            self.store.update(
                run_id,
                state="failed",
                stage="pipeline1",
                message="Pipeline 1 execution failed.",
                error={
                    "code": "pipeline1_execution_failed",
                    "message": "Pipeline 1 could not complete this dataset.",
                },
            )

    def status(self, run_id: str) -> dict[str, Any]:
        state = self.store.read(run_id)
        return {
            "run_id": state["application_run_id"],
            "state": state["state"],
            "stage": state["stage"],
            "message": state["message"],
            "created_at": state["created_at"],
            "updated_at": state["updated_at"],
            "original_filename": state["original_filename"],
            "stored_extension": state["stored_extension"],
            "pipeline1_internal_run_id": state.get("pipeline1_internal_run_id"),
            "error": state.get("error"),
        }

    @staticmethod
    def _csv_header(path: Path) -> list[str]:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            try:
                return [str(value) for value in next(csv.reader(handle))]
            except StopIteration as exc:
                raise ServiceError(500, "prepared_dataset_invalid", "Prepared dataset has no header.") from exc

    def prediction_context(self, run_id: str) -> dict[str, Any]:
        """Re-evaluate canonical prediction eligibility against current artifacts."""
        state = self.store.read(run_id)
        if state["state"] in ACTIVE_STATES:
            raise ServiceError(409, "pipeline1_not_complete", "Pipeline 1 is still running.")
        if state["state"] == "failed":
            raise ServiceError(409, "pipeline1_failed", "Pipeline 1 did not complete successfully.")

        paths = self.store.paths(run_id)
        report_path = paths["pipeline1"] / "cleaning_report.json"
        cleaned_path = paths["pipeline1"] / "cleaned_dataset.csv"
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ServiceError(
                500,
                "pipeline1_artifacts_unavailable",
                "Pipeline 1 result artifacts are unavailable or invalid.",
            ) from exc

        columns: list[str] = []
        columns_unique = True
        if cleaned_path.is_file():
            try:
                columns = self._csv_header(cleaned_path)
            except (OSError, UnicodeError, csv.Error) as exc:
                raise ServiceError(
                    500,
                    "pipeline1_artifacts_unavailable",
                    "Pipeline 1 result artifacts are unavailable or invalid.",
                ) from exc
            columns_unique = len(columns) == len(set(columns))

        return {
            "cleaned_path": cleaned_path,
            "target_columns": columns if columns_unique else [],
            # This is the canonical, run-scoped report used by backend
            # prediction execution.  It is intentionally not the sanitized
            # presentation copy returned by ``pipeline1_result``.
            "pipeline1_report": report,
            "eligibility": prediction_eligibility(
                report,
                cleaned_path,
                columns_unique=columns_unique,
            ),
        }

    def pipeline1_result(self, run_id: str) -> dict[str, Any]:
        state = self.store.read(run_id)
        if state["state"] in ACTIVE_STATES:
            raise ServiceError(409, "pipeline1_not_complete", "Pipeline 1 is still running.")
        if state["state"] == "failed":
            raise ServiceError(409, "pipeline1_failed", "Pipeline 1 did not complete successfully.")
        paths = self.store.paths(run_id)
        report_path = paths["pipeline1"] / "cleaning_report.json"
        cleaned_path = paths["pipeline1"] / "cleaned_dataset.csv"
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            columns = self._csv_header(cleaned_path)
            frame = pd.read_csv(cleaned_path, low_memory=False)
        except ServiceError:
            raise
        except (OSError, json.JSONDecodeError, pd.errors.ParserError) as exc:
            raise ServiceError(
                500, "pipeline1_artifacts_unavailable",
                "Pipeline 1 result artifacts are unavailable or invalid.",
            ) from exc
        columns_unique = len(columns) == len(set(columns)) and columns == [str(c) for c in frame.columns]
        eligibility = prediction_eligibility(
            report, cleaned_path, columns_unique=columns_unique
        )
        privacy = _as_mapping(
            _as_mapping(report.get("decision_summary")).get("privacy_actions")
        )
        return {
            "run_id": run_id,
            "state": "complete",
            "pipeline1_internal_run_id": str(report["run_id"]),
            "pipeline1_report": sanitize_pipeline1_report(report, run_id),
            "prepared_dataset": {
                "rows": int(len(frame)),
                "columns": int(len(columns)),
                "target_columns": columns if columns_unique else [],
            },
            "privacy_clearance_passed": bool(privacy.get("privacy_clearance_passed")),
            "prediction_eligibility": eligibility,
        }

    def cleanup_expired_runs(self) -> list[str]:
        return self.store.cleanup_expired(self.settings.run_retention_seconds)

    def shutdown(self, *, wait: bool = True) -> None:
        self.executor.shutdown(wait=wait, cancel_futures=False)


__all__ = [
    "ApplicationRunService",
    "RunStore",
    "ServiceError",
    "prediction_eligibility",
    "sanitize_display_filename",
    "sanitize_pipeline1_report",
]
