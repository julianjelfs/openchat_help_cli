#!/usr/bin/env python3
"""
Extract the OpenChat content and moderation guidelines into the corpus format.

The guidelines are a Svelte component built from numbered <CollapsibleCard>
sections. Each card is one chunk, and — unlike the blog posts — the page
supports per-section deep links (?section=N), so every guidelines citation
lands the reader on the exact rule rather than the top of a long page.

    frontend/app/src/components/landingpages/GuidelinesContent.svelte

The FAQ already points users here ("For full details of our content and
moderation guidelines see here"), so without this source the corpus has a
dangling reference: it can tell a user the rules exist but not what they say.

Usage:
    python ingest_guidelines.py --repo ./oc --out corpus/guidelines.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent))
from ingest_blog import node_to_text, preprocess, tidy

GUIDELINES_PATH = "frontend/app/src/components/landingpages/GuidelinesContent.svelte"
GUIDELINES_URL = "https://oc.app/guidelines"

TARGET_MAX_CHARS = 1500  # split above this, at paragraph boundaries


def git_meta(repo: Path, path: str) -> dict:
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", str(repo), *args],
                capture_output=True,
                text=True,
                check=True,
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


def resolve_helper_hrefs(src: str) -> str:
    """Rewrite href={getUrl("/x")} into a real URL.

    The blog preprocessor turns attr={expr} into attr="expr" verbatim, which
    would otherwise leak `getUrl("/terms")` into the corpus as if it were a
    link target.
    """
    return re.sub(
        r'href=\{getUrl\((["\'])(?P<path>[^"\']*)\1\)\}',
        lambda m: f'href="https://oc.app{m.group("path")}"',
        src,
    )


def parse_sections(src: str) -> list[dict]:
    """One entry per CollapsibleCard: its number, title and body text.

    The component renders the title inside a {#snippet titleSlot()} and the
    prose in a sibling div, so the section number in <span class="subtitle">
    is the anchor both for the chunk id and the ?section= deep link.
    """
    soup = BeautifulSoup(preprocess(resolve_helper_hrefs(src)), "html.parser")
    sections = []

    for card in soup.find_all("CollapsibleCard".lower()) or soup.find_all("collapsiblecard"):
        subtitle = card.find(class_="subtitle")
        title_div = card.find(class_="title")
        body = card.find(class_="body")
        if not (subtitle and title_div and body):
            continue

        number = subtitle.get_text(strip=True)
        # The title div also holds the copy-link icon; take the leading text.
        title = tidy("".join(node_to_text(c) for c in title_div.children)).split("\n")[0].strip()
        text = tidy("".join(node_to_text(c) for c in body.children))
        if not text:
            continue
        sections.append({"number": number, "title": title, "text": text})

    return sections


def split_long(text: str, limit: int = TARGET_MAX_CHARS) -> list[str]:
    """Split at paragraph boundaries only. A rule cut mid-sentence is worse
    than a slightly oversized chunk."""
    if len(text) <= limit:
        return [text]
    parts, current = [], ""
    for para in text.split("\n\n"):
        if current and len(current) + len(para) + 2 > limit:
            parts.append(current.strip())
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current.strip():
        parts.append(current.strip())
    return parts


def build(repo: Path) -> list[dict]:
    src = (repo / GUIDELINES_PATH).read_text()
    sections = parse_sections(src)
    if not sections:
        raise SystemExit(
            f"no sections parsed from {GUIDELINES_PATH} — the component markup has "
            "probably changed; fix the parser rather than shipping an empty source"
        )

    provenance = git_meta(repo, GUIDELINES_PATH)
    extracted_at = datetime.now(UTC).isoformat()
    chunks = []

    for section in sections:
        parts = split_long(section["text"])
        for index, part in enumerate(parts):
            suffix = "" if len(parts) == 1 else f".{index}"
            breadcrumb = f"Guidelines > {section['number']}. {section['title']}"
            text = f"{breadcrumb}\n\n{part}"
            chunks.append(
                {
                    "id": f"guidelines:{section['number']}{suffix}",
                    "source_type": "guidelines",
                    "title": breadcrumb,
                    "text": text,
                    # Per-section deep link: the page reads ?section= on load.
                    "url": f"{GUIDELINES_URL}?section={section['number']}",
                    "meta": {
                        "section_number": int(section["number"]),
                        "section_title": section["title"],
                        "part": index,
                        "char_count": len(text),
                    },
                    "provenance": {**provenance, "extracted_at": extracted_at},
                    "content_hash": hashlib.sha256(text.encode()).hexdigest()[:16],
                }
            )
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("corpus/guidelines.jsonl"))
    args = ap.parse_args()

    chunks = build(args.repo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"{len(chunks)} chunks -> {args.out}")
    for chunk in chunks:
        print(f"  {chunk['id']:20} {chunk['meta']['char_count']:>5} chars  {chunk['title']}")


if __name__ == "__main__":
    main()
