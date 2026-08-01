"""The Q&A service (SPEC.md Phase 5). FastAPI, POST /ask, GET /health.

    uv run serve-ocqa

Contract highlights, enforced here rather than hoped for:

- Answer generation is Pydantic-validated with one retry, then a refusal —
  that lives in the answerer (`ocqa.answering`), not here.
- Every returned chunk_id must exist in the index. A fabricated citation is a
  500, never a response — it must not reach a user with our authority
  attached.
- ``refused: true`` responses carry an empty citation list and a pointer to
  the help channel.
- Structured logging: one JSON line per request with question, strategy,
  retrieved ids, cited ids, refusal and latency. These logs are the next eval
  set.

Default strategy is ``dense``: the measured Phase 3 winner. The spec named
``hybrid+rerank``, but Phase 4 was not built — dense already retrieves every
golden expected chunk at k=5, leaving hybrid nothing to demonstrate at this
corpus size (see README).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ocqa.answering import Answerer, RetrievalAnswerer, StubRefusalAnswerer, StuffedAnswerer
from ocqa.corpus import load_corpus
from ocqa.models import Chunk, Citation

DEFAULT_STRATEGY = "hybrid"

logger = logging.getLogger("ocqa.service")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    strategy: str = DEFAULT_STRATEGY
    max_chunks: int = Field(default=5, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    refused: bool
    confidence: float
    strategy: str
    latency_ms: int


class ServiceState:
    """Everything built once at startup: corpus, index, answerers."""

    def __init__(self, answerers: dict[str, Answerer], chunks: list[Chunk], index_build_ms: int):
        self.answerers = answerers
        self.chunk_by_id = {chunk.id: chunk for chunk in chunks}
        self.index_build_ms = index_build_ms


def build_state(
    corpus_dir: Path = Path("corpus"),
    answer_model: str = "gpt-5",
    embed_model: str | None = None,
    reasoning_effort: str | None = None,
) -> ServiceState:
    from openai import OpenAI

    from ocqa.retrieval import build_retriever

    started = time.perf_counter()
    chunks = load_corpus(corpus_dir)
    client = OpenAI()
    dense = build_retriever("dense", chunks, client, embed_model)
    hybrid = build_retriever("hybrid", chunks, client, embed_model)
    answerers: dict[str, Answerer] = {
        "hybrid": RetrievalAnswerer(
            client, hybrid, model=answer_model, reasoning_effort=reasoning_effort
        ),
        "dense": RetrievalAnswerer(
            client, dense, model=answer_model, reasoning_effort=reasoning_effort
        ),
        "stuffed": StuffedAnswerer(
            client, chunks, model=answer_model, reasoning_effort=reasoning_effort
        ),
        "stub": StubRefusalAnswerer(),
    }
    index_build_ms = int((time.perf_counter() - started) * 1000)
    return ServiceState(answerers, chunks, index_build_ms)


def create_app(state: ServiceState) -> FastAPI:
    app = FastAPI(title="ocqa", description="Cited Q&A over the OpenChat corpus")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "corpus_chunks": len(state.chunk_by_id),
            "index_build_ms": state.index_build_ms,
            "strategies": sorted(state.answerers),
        }

    @app.post("/ask")
    def ask(request: AskRequest) -> AskResponse:
        answerer = state.answerers.get(request.strategy)
        if answerer is None:
            raise HTTPException(
                status_code=422,
                detail=f"unknown strategy {request.strategy!r}; "
                f"available: {sorted(state.answerers)}",
            )
        if isinstance(answerer, RetrievalAnswerer) and request.max_chunks != answerer.max_chunks:
            # Cheap per-request override: shares the client, retriever and
            # embedding cache; only the k changes.
            answerer = RetrievalAnswerer(
                answerer._client,
                answerer._retriever,
                model=answerer.model,
                max_chunks=request.max_chunks,
                reasoning_effort=answerer.reasoning_effort,
            )

        started = time.perf_counter()
        answer = answerer.answer(request.question)
        latency_ms = int((time.perf_counter() - started) * 1000)

        unresolved = [cid for cid in answer.citations if cid not in state.chunk_by_id]
        logger.info(
            "%s",
            json.dumps(
                {
                    "ts": datetime.now(UTC).isoformat(),
                    "question": request.question,
                    "strategy": request.strategy,
                    "retrieved": answer.retrieved,
                    "cited": answer.citations,
                    "refused": answer.refused,
                    "confidence": answer.confidence,
                    "latency_ms": latency_ms,
                    "unresolved_citations": unresolved,
                },
                ensure_ascii=False,
            ),
        )
        if unresolved:
            # A fabricated citation must never reach a user (SPEC.md rule 5).
            raise HTTPException(
                status_code=500,
                detail=f"answer cited unknown chunk ids {unresolved}; refusing to respond",
            )

        return AskResponse(
            answer=answer.text,
            citations=[Citation.from_chunk(state.chunk_by_id[cid]) for cid in answer.citations],
            refused=answer.refused,
            confidence=answer.confidence,
            strategy=request.strategy,
            latency_ms=latency_ms,
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ocqa service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--corpus-dir", type=Path, default=Path("corpus"))
    # gpt-5-mini: measured equal to gpt-5 on this corpus after the rules-v2
    # tuning (see README), at ~1/5 the price. --answer-model gpt-5 to override.
    parser.add_argument("--answer-model", default="gpt-5-mini")
    parser.add_argument("--embed-model", default=None)
    parser.add_argument(
        "--reasoning-effort", choices=["minimal", "low", "medium", "high"], default=None
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    state = build_state(
        corpus_dir=args.corpus_dir,
        answer_model=args.answer_model,
        embed_model=args.embed_model,
        reasoning_effort=args.reasoning_effort,
    )
    app = create_app(state)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
