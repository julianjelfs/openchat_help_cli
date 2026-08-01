"""Core data models.

The Chunk model is the data contract from SPEC.md: every corpus line, from
every source, validates against it. Anything that does not is a broken ingest,
not a special case.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SourceType = Literal["faq", "blog", "guidelines", "terms", "help_channel"]


class Chunk(BaseModel):
    id: str = Field(min_length=1)
    source_type: SourceType
    title: str
    text: str = Field(min_length=1)
    url: str = Field(min_length=1)
    meta: dict = Field(default_factory=dict)
    provenance: dict = Field(default_factory=dict)
    content_hash: str = Field(min_length=1)
    # Only help_channel chunks carry a status; anything not explicitly
    # approved must never be indexed.
    status: str | None = None

    @property
    def indexable(self) -> bool:
        if self.source_type == "help_channel":
            return self.status == "approved"
        return True


class Answer(BaseModel):
    """The output of any answering strategy, stub or real."""

    text: str
    citations: list[str] = Field(default_factory=list)
    refused: bool
    confidence: float = Field(ge=0.0, le=1.0)
    strategy: str
    # The chunk ids that were put in front of the model (retrieval-backed
    # strategies only). Kept separate from citations so a wrong answer can be
    # diagnosed instantly as "never retrieved" vs "retrieved and misused".
    retrieved: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    """A resolved citation, as returned by the service."""

    chunk_id: str
    url: str
    title: str
    source_type: SourceType
    published: str | None = None

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> Citation:
        return cls(
            chunk_id=chunk.id,
            url=chunk.url,
            title=chunk.title,
            source_type=chunk.source_type,
            published=chunk.meta.get("published") or chunk.meta.get("answered_at"),
        )
