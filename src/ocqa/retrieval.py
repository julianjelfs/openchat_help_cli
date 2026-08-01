"""Retrieval interfaces and the Phase 1 stub retriever.

The stub exists so the evaluation harness has something to measure before any
real retrieval is built (SPEC.md Phase 1 acceptance). It is deliberately
naive — unweighted token overlap — and sets the floor that Phase 3+ must beat.
"""

from __future__ import annotations

import re
from typing import Protocol

from ocqa.models import Chunk

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


class Retriever(Protocol):
    name: str

    def retrieve(self, question: str, k: int = 10) -> list[tuple[Chunk, float]]: ...


class StubLexicalRetriever:
    """Unweighted token-overlap scoring. Deterministic, no dependencies.

    Score is the number of distinct question tokens present in the chunk.
    Ties break on chunk id so ordering is stable across runs.
    """

    name = "stub-lexical"

    def __init__(self, chunks: list[Chunk]):
        self._chunks = [(chunk, tokenize(chunk.text + " " + chunk.title)) for chunk in chunks]

    def retrieve(self, question: str, k: int = 10) -> list[tuple[Chunk, float]]:
        q_tokens = tokenize(question)
        scored = [
            (chunk, float(len(q_tokens & chunk_tokens))) for chunk, chunk_tokens in self._chunks
        ]
        scored.sort(key=lambda pair: (-pair[1], pair[0].id))
        return scored[:k]
