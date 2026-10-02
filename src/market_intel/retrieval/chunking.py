"""Split document text into overlapping chunks and build deterministic chunk IDs."""
from __future__ import annotations

import hashlib


def chunk_text(text: str, size: int = 1200, overlap: int = 150) -> list[str]:
    """Windows of about `size` characters, cut at spaces, each overlapping the previous by ~`overlap`."""
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(" ", start + size // 2, end)  # prefer a word boundary
            end = cut if cut != -1 else end
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_id(doc_id: str, chunk_idx: int) -> str:
    """Deterministic vector ID, so re-ingesting a document overwrites its vectors instead of adding new ones."""
    return f"{doc_id}:{chunk_idx}"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
