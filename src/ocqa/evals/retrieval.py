"""Retrieval eval. Deterministic, no LLM, runs in seconds.

    uv run eval-retrieval

Scores the retriever against golden ``expected_chunk_ids``: recall@1/3/5/10
and MRR, broken down per category and per source_type. Only cases that carry
expected chunk ids are scored (refusal/ambiguous/injection cases generally
have none — there is nothing correct to retrieve for them).

The per-source_type breakdown is a micro average over expected chunk ids of
that type ("of all the faq chunks we expected, how many were in the top k"),
because a single case can expect chunks from several sources.

Results go to evals/results/<timestamp>-retrieval.json and a summary to
stdout.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from ocqa.corpus import load_corpus
from ocqa.embeddings import DEFAULT_EMBED_MODEL, OpenAIEmbedder
from ocqa.evals.golden import load_golden, validate_against_corpus
from ocqa.evals.metrics import mean, recall_at_k, reciprocal_rank
from ocqa.retrieval import DenseRetriever, StubLexicalRetriever

K_VALUES = (1, 3, 5, 10)


def run_eval(retriever, cases, chunks) -> dict:
    chunk_by_id = {chunk.id: chunk for chunk in chunks}
    scored_cases = [case for case in cases if case.expected_chunk_ids]

    per_case = []
    for case in scored_cases:
        retrieved = [chunk.id for chunk, _ in retriever.retrieve(case.question, k=max(K_VALUES))]
        per_case.append(
            {
                "case_id": case.id,
                "category": case.category,
                "expected": case.expected_chunk_ids,
                "retrieved": retrieved,
                "recall": {
                    str(k): recall_at_k(case.expected_chunk_ids, retrieved, k) for k in K_VALUES
                },
                "reciprocal_rank": reciprocal_rank(case.expected_chunk_ids, retrieved),
            }
        )

    def summarise(rows: list[dict]) -> dict:
        return {
            "cases": len(rows),
            **{
                f"recall@{k}": round(mean([row["recall"][str(k)] for row in rows]), 4)
                for k in K_VALUES
            },
            "mrr": round(mean([row["reciprocal_rank"] for row in rows]), 4),
        }

    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in per_case:
        by_category[row["category"]].append(row)

    # Micro average per source_type of the expected chunks.
    by_source: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in per_case:
        for chunk_id in row["expected"]:
            source = chunk_by_id[chunk_id].source_type
            for k in K_VALUES:
                by_source[source][f"recall@{k}"].append(
                    1.0 if chunk_id in row["retrieved"][:k] else 0.0
                )

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "retriever": retriever.name,
        "embed_model": getattr(retriever, "embed_model", None),
        "corpus_chunks": len(chunks),
        "golden_cases_total": len(cases),
        "golden_cases_scored": len(scored_cases),
        "overall": summarise(per_case),
        "by_category": {cat: summarise(rows) for cat, rows in sorted(by_category.items())},
        "by_source_type": {
            source: {
                "expected_chunks": len(values[f"recall@{K_VALUES[0]}"]),
                **{key: round(mean(vals), 4) for key, vals in sorted(values.items())},
            }
            for source, values in sorted(by_source.items())
        },
        "per_case": per_case,
    }


def print_summary(report: dict) -> None:
    print(f"\nretrieval eval — {report['retriever']}")
    print(
        f"corpus: {report['corpus_chunks']} chunks, scored cases: {report['golden_cases_scored']}"
    )
    header = f"{'':<16}" + "".join(f"{f'r@{k}':>8}" for k in K_VALUES) + f"{'mrr':>8}{'n':>6}"
    print(header)

    def line(label: str, stats: dict) -> str:
        return (
            f"{label:<16}"
            + "".join(f"{stats[f'recall@{k}']:>8.3f}" for k in K_VALUES)
            + f"{stats['mrr']:>8.3f}{stats['cases']:>6}"
        )

    print(line("overall", report["overall"]))
    for category, stats in report["by_category"].items():
        print(line(f"  {category}", stats))
    print("\nby source_type (micro, over expected chunks):")
    for source, stats in report["by_source_type"].items():
        recalls = "".join(f"{stats[f'recall@{k}']:>8.3f}" for k in K_VALUES)
        print(f"{source:<16}{recalls}{'':>8}{stats['expected_chunks']:>6}")

    worst = sorted(report["per_case"], key=lambda row: row["reciprocal_rank"])[:5]
    print("\nworst cases by reciprocal rank:")
    for row in worst:
        print(f"  {row['case_id']}  rr={row['reciprocal_rank']:.3f}  expected={row['expected']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic retrieval eval")
    parser.add_argument("--strategy", choices=["stub", "dense"], default="dense")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--corpus-dir", type=Path, default=Path("corpus"))
    parser.add_argument("--golden", type=Path, default=Path("evals/golden.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("evals/results"))
    args = parser.parse_args()

    chunks = load_corpus(args.corpus_dir)
    cases = load_golden(args.golden)
    validate_against_corpus(cases, chunks)

    if args.strategy == "dense":
        # Embedding calls happen only on cache misses; with a warm cache this
        # command still runs offline in seconds.
        from openai import OpenAI

        retriever = DenseRetriever(chunks, OpenAIEmbedder(OpenAI(), model=args.embed_model))
    else:
        retriever = StubLexicalRetriever(chunks)
    report = run_eval(retriever, cases, chunks)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out_dir / f"{stamp}-retrieval.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")

    print_summary(report)
    print(f"\nfull report: {out_path}")


if __name__ == "__main__":
    main()
