"""Retrieval metrics. Pure functions, unit-tested in isolation.

recall@k here is the fraction of a case's expected chunks found in the top k,
averaged over cases (macro average). MRR uses the rank of the first expected
chunk to appear.
"""

from __future__ import annotations


def recall_at_k(expected: list[str], retrieved: list[str], k: int) -> float:
    if not expected:
        raise ValueError("recall is undefined for a case with no expected chunks")
    top_k = set(retrieved[:k])
    return sum(1 for chunk_id in expected if chunk_id in top_k) / len(expected)


def reciprocal_rank(expected: list[str], retrieved: list[str]) -> float:
    expected_set = set(expected)
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in expected_set:
            return 1.0 / rank
    return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
