"""Retrieval metrics. Pure functions, unit-tested in isolation.

recall@k here is the fraction of a case's expected chunks found in the top k,
averaged over cases (macro average). MRR uses the rank of the first expected
chunk to appear.

An expected entry may be a single chunk id, or a list of ids meaning "any one
of these is correct". The corpus deliberately carries the same rule in more
than one place — the terms schedules restate the guidelines — and retrieving
either is a success, not half a success.
"""

from __future__ import annotations

Expected = str | list[str]


def _alternates(entry: Expected) -> set[str]:
    return {entry} if isinstance(entry, str) else set(entry)


def flatten(expected: list[Expected]) -> set[str]:
    """Every id that would satisfy any part of the expectation."""
    return {chunk_id for entry in expected for chunk_id in _alternates(entry)}


def recall_at_k(expected: list[Expected], retrieved: list[str], k: int) -> float:
    if not expected:
        raise ValueError("recall is undefined for a case with no expected chunks")
    top_k = set(retrieved[:k])
    hits = sum(1 for entry in expected if _alternates(entry) & top_k)
    return hits / len(expected)


def reciprocal_rank(expected: list[Expected], retrieved: list[str]) -> float:
    acceptable = flatten(expected)
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in acceptable:
            return 1.0 / rank
    return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
