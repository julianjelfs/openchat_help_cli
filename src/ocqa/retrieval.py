"""Retrieval strategies.

- ``stub-lexical`` (Phase 1): unweighted token overlap. The floor.
- ``dense`` (Phase 3): embed every chunk, brute-force cosine over a single
  numpy matrix. Deliberately unsophisticated — 100 chunks needs nothing more.
"""

from __future__ import annotations

import re
from typing import Protocol

import numpy as np

from ocqa.embeddings import OpenAIEmbedder, text_key
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


class DenseRetriever:
    """Brute-force cosine similarity over embedded chunks (SPEC.md Phase 3).

    Chunk ``text`` is what gets embedded (the data contract says it already
    carries its own context). Vectors are L2-normalised at build time so
    retrieval is a single matrix-vector product.
    """

    name = "dense"

    def __init__(self, chunks: list[Chunk], embedder: OpenAIEmbedder):
        self._chunks = chunks
        self._embedder = embedder
        self.embed_model = embedder.model
        matrix = embedder.embed([(chunk.content_hash, chunk.text) for chunk in chunks])
        self._matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

    def retrieve(self, question: str, k: int = 10) -> list[tuple[Chunk, float]]:
        vector = self._embedder.embed([(text_key(question), question)])[0]
        vector = vector / np.linalg.norm(vector)
        scores = self._matrix @ vector
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self._chunks[i], float(scores[i])) for i in order]
