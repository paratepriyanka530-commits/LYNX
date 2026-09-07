import re
from typing import Any, Dict, List


def chunk_knowledge_document(
    document: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Split a LYNX knowledge document by Markdown H2 sections.

    Each chunk retains the metadata required for retrieval,
    traceability, and later RAG citations.
    """

    content = document["content"]

    content = re.sub(
        r"^#\s+.+?\n+",
        "",
        content,
        count=1
    )

    # Split on Markdown H2 headings while retaining the heading.
    sections = re.split(
        r"(?=^##\s+)",
        content,
        flags=re.MULTILINE
    )

    chunks = []

    for index, section in enumerate(sections):
        section = section.strip()

        if not section:
            continue

        # Ignore the document-level H1 if it contains no
        # useful knowledge beyond the title.
        if (
            section.startswith("# ")
            and "\n" not in section
        ):
            continue

        lines = section.splitlines()

        if lines[0].startswith("## "):
            section_title = lines[0][3:].strip()
            section_content = "\n".join(
                lines[1:]
            ).strip()
        else:
            section_title = "Overview"
            section_content = section

        if not section_content:
            continue

        chunk = {
            "chunk_id": (
                f"{document['knowledge_id']}"
                f"-{index:03d}"
            ),
            "knowledge_id": document["knowledge_id"],
            "category": document["category"],
            "topic": document["topic"],
            "version": document["version"],
            "section": section_title,
            "content": section_content,
            "source": document["source"],
        }

        chunks.append(chunk)

    return chunks