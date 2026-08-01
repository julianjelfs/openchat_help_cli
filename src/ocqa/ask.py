"""One-shot question answering from the command line.

    uv run ask "How do I buy CHAT tokens?"

The same library path the service uses, without the HTTP hop: load the
corpus, retrieve, answer, print the answer with resolvable citations. Useful
for spot-checking the system by hand, which the eval harness deliberately
does not replace — reading a few real answers catches things a score does
not.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ocqa.answering import RetrievalAnswerer, StuffedAnswerer
from ocqa.corpus import load_corpus
from ocqa.models import Citation


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the OpenChat corpus a question")
    parser.add_argument("question", help="the question to answer")
    parser.add_argument(
        "--strategy", choices=["dense", "bm25", "hybrid", "stuffed"], default="dense"
    )
    parser.add_argument("--max-chunks", type=int, default=5)
    parser.add_argument("--answer-model", default="gpt-5-mini")
    parser.add_argument("--embed-model", default=None)
    parser.add_argument("--corpus-dir", type=Path, default=Path("corpus"))
    parser.add_argument("--json", action="store_true", help="emit the raw response object")
    parser.add_argument(
        "--show-retrieved",
        action="store_true",
        help="also list what retrieval put in front of the model, cited or not",
    )
    args = parser.parse_args()

    if not (Path(args.corpus_dir).exists()):
        sys.exit(f"no corpus at {args.corpus_dir} — run from the repo root")

    from openai import OpenAI

    from ocqa.retrieval import build_retriever

    client = OpenAI(timeout=180.0)
    chunks = load_corpus(args.corpus_dir)
    chunk_by_id = {chunk.id: chunk for chunk in chunks}

    if args.strategy == "stuffed":
        answerer = StuffedAnswerer(client, chunks, model=args.answer_model)
    else:
        retriever = build_retriever(args.strategy, chunks, client, args.embed_model)
        answerer = RetrievalAnswerer(
            client, retriever, model=args.answer_model, max_chunks=args.max_chunks
        )

    started = time.perf_counter()
    answer = answerer.answer(args.question)
    latency_ms = int((time.perf_counter() - started) * 1000)

    citations = [Citation.from_chunk(chunk_by_id[cid]) for cid in answer.citations]

    if args.json:
        print(
            json.dumps(
                {
                    "question": args.question,
                    "answer": answer.text,
                    "citations": [citation.model_dump() for citation in citations],
                    "refused": answer.refused,
                    "confidence": answer.confidence,
                    "strategy": args.strategy,
                    "retrieved": answer.retrieved,
                    "latency_ms": latency_ms,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    print(f"\n{answer.text}\n")

    if citations:
        print("Sources:")
        for index, citation in enumerate(citations, start=1):
            published = f", {citation.published[:10]}" if citation.published else ""
            print(f"  [{index}] {citation.title}")
            print(f"      {citation.url}  ({citation.source_type}{published})")
    elif not answer.refused:
        print("Sources: none cited")

    if args.show_retrieved:
        print("\nRetrieved (what the model was shown):")
        for chunk_id in answer.retrieved:
            marker = "*" if chunk_id in answer.citations else " "
            print(f"  {marker} {chunk_id}")

    state = "refused" if answer.refused else f"confidence {answer.confidence:.2f}"
    print(f"\n[{args.strategy} · {state} · {latency_ms}ms]")


if __name__ == "__main__":
    main()
