from pathlib import Path
from typing import Any, Dict

import yaml


def load_knowledge_document(
    file_path: str
) -> Dict[str, Any]:
    """
    Load a LYNX knowledge-base Markdown document.

    Expected format:

    ---
    knowledge_id: SEM-001
    category: semantic
    topic: identifier
    version: 1.0
    ---

    Markdown content...
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Knowledge document not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Knowledge path is not a file: {path}"
        )

    if path.suffix.lower() != ".md":
        raise ValueError(
            f"Unsupported knowledge document format: {path.suffix}"
        )

    text = path.read_text(
        encoding="utf-8"
    )

    if not text.strip():
        raise ValueError(
            f"Knowledge document is empty: {path.name}"
        )

    # Separate YAML front matter from Markdown body.
    if not text.startswith("---"):
        raise ValueError(
            f"Missing YAML front matter: {path.name}"
        )

    parts = text.split("---", 2)

    if len(parts) < 3:
        raise ValueError(
            f"Invalid YAML front matter: {path.name}"
        )

    metadata_text = parts[1].strip()
    content = parts[2].strip()

    metadata = yaml.safe_load(
        metadata_text
    )

    if not isinstance(metadata, dict):
        raise ValueError(
            f"Invalid metadata in: {path.name}"
        )

    required_fields = {
        "knowledge_id",
        "category",
        "topic",
        "version",
    }

    missing_fields = (
        required_fields - metadata.keys()
    )

    if missing_fields:
        raise ValueError(
            "Missing required metadata fields: "
            + ", ".join(sorted(missing_fields))
        )

    if not content:
        raise ValueError(
            f"Knowledge content is empty: {path.name}"
        )

    return {
        "knowledge_id": str(
            metadata["knowledge_id"]
        ),
        "category": str(
            metadata["category"]
        ),
        "topic": str(
            metadata["topic"]
        ),
        "version": str(
            metadata["version"]
        ),
        "content": content,
        "source": str(path),
    }