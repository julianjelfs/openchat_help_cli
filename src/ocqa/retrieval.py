"""Retrieval strategies.

- ``stub-lexical`` (Phase 1): unweighted token overlap. The floor.
- ``dense`` (Phase 3): embed every chunk, brute-force cosine over a single
  numpy matrix. Deliberately unsophisticated — 100 chunks needs nothing more.
- ``bm25`` / ``hybrid`` (Phase 4.1): Okapi BM25, and reciprocal rank fusion
  of BM25 with dense. The corpus is thick with product jargon (canister,
  chit, CHAT, ICP, neuron, Diamond) and formal legal phrasing, where exact
  term matching is the thing embeddings are worst at.

BM25 is ~40 lines of numpy here rather than a dependency: at this corpus
size the library would be more indirection than code.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol

import numpy as np

from ocqa.embeddings import OpenAIEmbedder, content_key, text_key
from ocqa.models import Chunk

_TOKEN = re.compile(r"[a-z0-9]+")

BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60  # standard reciprocal-rank-fusion damping


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def tokenize(text: str) -> set[str]:
    return set(tokens(text))


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
        matrix = embedder.embed([(content_key(chunk.text), chunk.text) for chunk in chunks])
        self._matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

    def retrieve(self, question: str, k: int = 10) -> list[tuple[Chunk, float]]:
        vector = self._embedder.embed([(text_key(question), question)])[0]
        vector = vector / np.linalg.norm(vector)
        scores = self._matrix @ vector
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self._chunks[i], float(scores[i])) for i in order]


class BM25Retriever:
    """Okapi BM25 over chunk text.

    Scores exact term overlap with saturation (k1) and length normalisation
    (b), which is what dense embeddings are weakest at: acronyms, identifiers
    and the formal vocabulary of the terms of use.
    """

    name = "bm25"

    def __init__(self, chunks: list[Chunk], k1: float = BM25_K1, b: float = BM25_B):
        self._chunks = chunks
        self.k1, self.b = k1, b
        self._docs = [Counter(tokens(f"{chunk.title} {chunk.text}")) for chunk in chunks]
        self._lengths = np.array([sum(doc.values()) for doc in self._docs], dtype=np.float32)
        self._avg_length = float(self._lengths.mean()) if len(self._lengths) else 0.0

        document_frequency = Counter()
        for doc in self._docs:
            document_frequency.update(doc.keys())
        total = len(self._docs)
        # Robertson/Sparck-Jones idf with the +0.5 smoothing, floored at zero
        # so that a term in every document cannot push scores negative.
        self._idf = {
            term: max(0.0, math.log(1 + (total - freq + 0.5) / (freq + 0.5)))
            for term, freq in document_frequency.items()
        }

    def retrieve(self, question: str, k: int = 10) -> list[tuple[Chunk, float]]:
        query_terms = tokens(question)
        scores = np.zeros(len(self._chunks), dtype=np.float32)
        for term in query_terms:
            idf = self._idf.get(term)
            if idf is None:
                continue
            for index, doc in enumerate(self._docs):
                freq = doc.get(term)
                if not freq:
                    continue
                norm = 1 - self.b + self.b * (self._lengths[index] / self._avg_length)
                scores[index] += idf * (freq * (self.k1 + 1)) / (freq + self.k1 * norm)

        order = np.argsort(-scores, kind="stable")[:k]
        return [(self._chunks[i], float(scores[i])) for i in order]


class HybridRetriever:
    """Reciprocal rank fusion of dense and BM25 rankings.

    RRF combines by *rank*, not score, so the two retrievers' incomparable
    scales (cosine similarity vs BM25 magnitude) never need calibrating. Each
    list contributes 1/(RRF_K + rank); a chunk both agree on outranks one that
    either alone puts first.

    Candidates are pulled deeper than k from each retriever (``depth``) so a
    chunk ranked, say, 12th by dense and 3rd by BM25 can still surface.
    """

    name = "hybrid"

    def __init__(self, dense: Retriever, lexical: Retriever, depth: int = 30, rrf_k: int = RRF_K):
        self._dense = dense
        self._lexical = lexical
        self.depth = depth
        self.rrf_k = rrf_k
        self.embed_model = getattr(dense, "embed_model", None)

    def retrieve(self, question: str, k: int = 10) -> list[tuple[Chunk, float]]:
        rankings = [
            self._dense.retrieve(question, k=self.depth),
            self._lexical.retrieve(question, k=self.depth),
        ]
        fused: dict[str, float] = {}
        chunk_by_id: dict[str, Chunk] = {}
        for ranking in rankings:
            for rank, (chunk, _) in enumerate(ranking, start=1):
                chunk_by_id[chunk.id] = chunk
                fused[chunk.id] = fused.get(chunk.id, 0.0) + 1.0 / (self.rrf_k + rank)

        ordered = sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))
        return [(chunk_by_id[chunk_id], score) for chunk_id, score in ordered[:k]]


def build_retriever(name: str, chunks: list[Chunk], client=None, embed_model: str | None = None):
    """Construct a retriever by strategy name.

    Kept in one place so the evals and the service cannot drift apart on what
    "hybrid" means.
    """
    if name == "stub":
        return StubLexicalRetriever(chunks)
    if name == "bm25":
        return BM25Retriever(chunks)

    from ocqa.embeddings import DEFAULT_EMBED_MODEL

    embedder = OpenAIEmbedder(client, model=embed_model or DEFAULT_EMBED_MODEL)
    dense = DenseRetriever(chunks, embedder)
    if name == "dense":
        return dense
    if name == "hybrid":
        return HybridRetriever(dense, BM25Retriever(chunks))
    raise ValueError(f"unknown retrieval strategy {name!r}")
