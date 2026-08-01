"""Add a hand-written question and answer to the corpus.

    uv run add-answer -q "Can I change my username?" \\
        -a "Yes — open profile settings and edit the username field." \\
        -u https://oc.app/faq

Writes one validated line to ``corpus/manual.jsonl``. That file is an
ordinary corpus source: the loader picks up every ``*.jsonl`` in the corpus
directory, so a new entry is retrievable on the next run with no other step.

Hand-editing the file directly also works — this exists so the fiddly parts
(id uniqueness, the content hash, schema validity) cannot be got wrong
silently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from ocqa.models import Chunk

DEFAULT_PATH = Path("corpus/manual.jsonl")


def slugify(text: str, limit: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:limit] or "entry"


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()}


def build_chunk(question: str, answer: str, url: str, author: str | None, chunk_id: str) -> Chunk:
    text = f"{question}\n\n{answer}"
    return Chunk(
        id=chunk_id,
        source_type="manual",
        title=question,
        text=text,
        url=url,
        meta={
            "question": question,
            "answer": answer,
            "author": author,
            "added_at": datetime.now(UTC).isoformat(),
        },
        provenance={"source": "hand-written", "added_at": datetime.now(UTC).isoformat()},
        content_hash=hashlib.sha256(text.encode()).hexdigest()[:16],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Add a hand-written Q&A to the corpus")
    parser.add_argument("-q", "--question", required=True)
    parser.add_argument("-a", "--answer", required=True)
    parser.add_argument(
        "-u",
        "--url",
        default="https://oc.app/faq",
        help="where a reader can verify this; shown in citations",
    )
    parser.add_argument("--id", default=None, help="defaults to manual:<slug of question>")
    parser.add_argument("--author", default=None, help="recorded in meta, for provenance")
    parser.add_argument("--out", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--force", action="store_true", help="allow replacing an existing id")
    args = parser.parse_args()

    chunk_id = args.id or f"manual:{slugify(args.question)}"
    known = existing_ids(args.out)
    if chunk_id in known and not args.force:
        sys.exit(
            f"{chunk_id} already exists in {args.out}. "
            "Pass --id to choose another, or --force to append a replacement."
        )

    try:
        chunk = build_chunk(args.question, args.answer, args.url, args.author, chunk_id)
    except ValidationError as err:
        sys.exit(f"invalid entry: {err}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as fh:
        fh.write(chunk.model_dump_json(exclude_none=False) + "\n")

    print(f"added {chunk.id} to {args.out}")
    print(f"  {chunk.title}")
    print("\nIt is indexed on the next run — try:")
    print(f'  uv run ask "{args.question}"')
    print("\nIf this answers a question the corpus previously could not, add a")
    print("golden case for it (and flip any refusal case it now covers).")


if __name__ == "__main__":
    main()
