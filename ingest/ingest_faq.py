#!/usr/bin/env python3
"""
Extract the OpenChat FAQs into the flat corpus format.

Source of truth is the i18n bundle in the open-chat repo, not the rendered
page: /faq is client-rendered, and the repo gives stable keys and git history.

    frontend/openchat-shared/src/domain/faq.ts   -> ordered list of question keys
    frontend/app/src/i18n/en.json                -> faq.<key>_q / faq.<key>_a

Usage:
    python ingest_faq.py --repo ./oc --out corpus/faq.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

FAQ_TS = "frontend/openchat-shared/src/domain/faq.ts"
EN_JSON = "frontend/app/src/i18n/en.json"
FAQ_BASE_URL = "https://oc.app/faq"

# The FAQ answers are markdown but contain some raw inline HTML anchors.
ANCHOR_RE = re.compile(
    r"""<a\b[^>]*?href=["'](?P<href>[^"']+)["'][^>]*>(?P<label>.*?)</a>""",
    re.IGNORECASE | re.DOTALL,
)
TAG_RE = re.compile(r"<[^>]+>")
MD_LINK_RE = re.compile(r"\[(?P<label>[^\]]*)\]\((?P<href>[^)\s]+)[^)]*\)")
WS_RE = re.compile(r"[ \t]+")
BLANKS_RE = re.compile(r"\n{3,}")


@dataclass
class Chunk:
    """One retrievable unit. Every source type lands in this shape."""

    id: str
    source_type: str  # faq | blog | help_channel
    title: str
    text: str  # what gets embedded
    url: str  # what gets cited
    meta: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    content_hash: str = ""

    def finalise(self) -> "Chunk":
        self.content_hash = hashlib.sha256(self.text.encode()).hexdigest()[:16]
        return self


def parse_question_keys(repo: Path) -> list[str]:
    """Read the ordered key list out of faq.ts rather than sorting en.json."""
    src = (repo / FAQ_TS).read_text()
    block = re.search(r"allQuestions\s*=\s*\[(.*?)\]", src, re.DOTALL)
    if not block:
        raise SystemExit(f"could not find allQuestions in {FAQ_TS}")
    return re.findall(r'"([^"]+)"', block.group(1))


def html_to_markdown(text: str) -> tuple[str, list[dict]]:
    """Fold inline <a> tags into markdown links; drop any other stray tags.

    Returns the cleaned text plus every outbound link found, so the retrieval
    layer can surface 'further reading' alongside the citation.
    """
    links: list[dict] = []

    def replace(m: re.Match) -> str:
        label = TAG_RE.sub("", m.group("label")).strip()
        href = m.group("href").strip()
        links.append({"label": label, "url": href})
        return f"[{label}]({href})"

    cleaned = ANCHOR_RE.sub(replace, text)
    cleaned = TAG_RE.sub("", cleaned)

    for m in MD_LINK_RE.finditer(cleaned):
        entry = {"label": m.group("label").strip(), "url": m.group("href")}
        if entry not in links:
            links.append(entry)

    cleaned = WS_RE.sub(" ", cleaned)
    cleaned = BLANKS_RE.sub("\n\n", cleaned)
    return cleaned.strip(), links


def git_meta(repo: Path, path: str) -> dict:
    """Commit and date for the source file.

    NB: a shallow clone only knows about HEAD, so this is the extraction
    baseline rather than a true per-entry last-modified. Clone with full
    history if you want real staleness signals per FAQ.
    """
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", str(repo), *args],
                capture_output=True, text=True, check=True,
            )
            return out.stdout.strip() or None
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    return {
        "repo": "open-chat-labs/open-chat",
        "path": path,
        "commit": run("rev-parse", "HEAD"),
        "commit_date": run("log", "-1", "--format=%cI", "--", path),
        "shallow": (repo / ".git" / "shallow").exists(),
    }


def build(repo: Path) -> list[Chunk]:
    keys = parse_question_keys(repo)
    strings = json.loads((repo / EN_JSON).read_text())["faq"]
    src_meta = git_meta(repo, EN_JSON)
    extracted_at = datetime.now(timezone.utc).isoformat()

    chunks: list[Chunk] = []
    for position, key in enumerate(keys):
        q_raw, a_raw = strings.get(f"{key}_q"), strings.get(f"{key}_a")
        if not q_raw or not a_raw:
            print(f"  ! skipping '{key}': missing question or answer")
            continue

        question, q_links = html_to_markdown(q_raw)
        answer, a_links = html_to_markdown(a_raw)

        chunks.append(
            Chunk(
                id=f"faq:{key}",
                source_type="faq",
                title=question,
                # Question text is prepended deliberately: user queries look far
                # more like the question than like the prose of the answer.
                text=f"{question}\n\n{answer}",
                url=f"{FAQ_BASE_URL}?q={key}",
                meta={
                    "faq_key": key,
                    "position": position,
                    "question": question,
                    "answer": answer,
                    "links": q_links + a_links,
                    "char_count": len(answer),
                },
                provenance={**src_meta, "extracted_at": extracted_at},
            ).finalise()
        )
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("oc"))
    ap.add_argument("--out", type=Path, default=Path("corpus/faq.jsonl"))
    args = ap.parse_args()

    chunks = build(args.repo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for chunk in chunks:
            fh.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    sizes = [len(c.text) for c in chunks]
    print(f"wrote {len(chunks)} chunks -> {args.out}")
    print(f"chars: min {min(sizes)}  max {max(sizes)}  mean {sum(sizes)//len(sizes)}")


if __name__ == "__main__":
    main()
