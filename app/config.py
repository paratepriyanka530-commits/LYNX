"""Configuration for the local LYNX application integration layer."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


@dataclass(frozen=True)
class ApplicationSettings:
    """Small, environment-overridable settings set for the single-machine app."""

    runtime_root: Path = PROJECT_ROOT / "runtime"
    maximum_upload_bytes: int = 50 * 1024 * 1024
    pipeline1_worker_count: int = 1
    prediction_worker_count: int = 1
    run_retention_seconds: int = 7 * 24 * 60 * 60
    upload_chunk_bytes: int = 1024 * 1024
    xlsx_max_entries: int = 10_000
    xlsx_max_uncompressed_bytes: int = 250 * 1024 * 1024
    xlsx_max_compression_ratio: int = 200

    @classmethod
    def from_environment(cls) -> "ApplicationSettings":
        runtime = Path(
            os.getenv("LYNX_RUNTIME_ROOT", str(PROJECT_ROOT / "runtime"))
        ).expanduser()
        return cls(
            runtime_root=runtime,
            maximum_upload_bytes=_positive_int(
                "LYNX_MAX_UPLOAD_BYTES", 50 * 1024 * 1024
            ),
            pipeline1_worker_count=_positive_int("LYNX_PIPELINE1_WORKERS", 1),
            prediction_worker_count=_positive_int("LYNX_PREDICTION_WORKERS", 1),
            run_retention_seconds=_positive_int(
                "LYNX_RUN_RETENTION_SECONDS", 7 * 24 * 60 * 60
            ),
            upload_chunk_bytes=_positive_int(
                "LYNX_UPLOAD_CHUNK_BYTES", 1024 * 1024
            ),
            xlsx_max_entries=_positive_int("LYNX_XLSX_MAX_ENTRIES", 10_000),
            xlsx_max_uncompressed_bytes=_positive_int(
                "LYNX_XLSX_MAX_UNCOMPRESSED_BYTES", 250 * 1024 * 1024
            ),
            xlsx_max_compression_ratio=_positive_int(
                "LYNX_XLSX_MAX_COMPRESSION_RATIO", 200
            ),
        )


__all__ = ["ApplicationSettings", "PROJECT_ROOT"]
