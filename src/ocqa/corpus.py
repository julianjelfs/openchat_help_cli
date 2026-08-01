"""Corpus loading.

Loads every ``*.jsonl`` file under the corpus directory, validates each line
against the Chunk contract, and filters out anything not indexable (i.e.
help-channel chunks that a human has not approved). Duplicate ids are a hard
error: citations reference ids, so an ambiguous id would make citations
unresolvable.
"""

from __future__ import annotations

from pathlib import Path

from ocqa.models import Chunk

DEFAULT_CORPUS_DIR = Path("corpus")


class CorpusError(Exception):
    pass


def load_corpus(corpus_dir: Path = DEFAULT_CORPUS_DIR) -> list[Chunk]:
    paths = sorted(corpus_dir.glob("*.jsonl"))
    if not paths:
        raise CorpusError(f"no *.jsonl files found in {corpus_dir}")

    chunks: list[Chunk] = []
    seen: set[str] = set()
    for path in paths:
        for line_no, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            chunk = Chunk.model_validate_json(line)
            if not chunk.indexable:
                continue
            if chunk.id in seen:
                raise CorpusError(f"duplicate chunk id {chunk.id!r} at {path}:{line_no}")
            seen.add(chunk.id)
            chunks.append(chunk)
    return chunks
