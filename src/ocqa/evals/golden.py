"""Golden set loading and validation.

``expected_chunk_ids`` is the retrieval ground truth. An id that does not
resolve against the corpus is a hard error (SPEC.md: citations must resolve —
the same discipline applies to the ground truth itself).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ocqa.models import Chunk

DEFAULT_GOLDEN_PATH = Path("evals/golden.jsonl")

Category = Literal["answerable", "refusal", "ambiguous", "injection"]

CATEGORY_MINIMUMS: dict[str, int] = {
    "answerable": 25,
    "refusal": 8,
    "ambiguous": 4,
    "injection": 3,
}


class GoldenCase(BaseModel):
    id: str = Field(pattern=r"^g\d{3}$")
    category: Category
    question: str = Field(min_length=1)
    expected_chunk_ids: list[str] = Field(default_factory=list)
    must_mention: list[str] = Field(default_factory=list)
    must_not_mention: list[str] = Field(default_factory=list)
    notes: str = ""


class GoldenError(Exception):
    pass


def load_golden(path: Path = DEFAULT_GOLDEN_PATH) -> list[GoldenCase]:
    cases: list[GoldenCase] = []
    seen: set[str] = set()
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        case = GoldenCase.model_validate_json(line)
        if case.id in seen:
            raise GoldenError(f"duplicate case id {case.id!r} at {path}:{line_no}")
        seen.add(case.id)
        cases.append(case)
    return cases


def validate_against_corpus(cases: list[GoldenCase], chunks: list[Chunk]) -> None:
    """Every expected chunk id must resolve to a real, indexable chunk."""
    known = {chunk.id for chunk in chunks}
    bad = [
        (case.id, chunk_id)
        for case in cases
        for chunk_id in case.expected_chunk_ids
        if chunk_id not in known
    ]
    if bad:
        detail = ", ".join(f"{case_id} -> {chunk_id}" for case_id, chunk_id in bad)
        raise GoldenError(f"expected_chunk_ids that do not resolve against the corpus: {detail}")
