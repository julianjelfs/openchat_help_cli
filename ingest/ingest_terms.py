#!/usr/bin/env python3
"""
Extract the OpenChat terms of use into the corpus format.

    frontend/app/src/components/landingpages/TermsContent.svelte

The guidelines page links to the terms and stops, the same dangling reference
the guidelines themselves used to be. This is a long legal document, so two
things matter more here than in the other sources:

1. Clause numbers. The markup carries none: the visible "A1", "1.1)" markers
   come from CSS counters (`content: var(--prefix) counter(item) var(--suffix)`)
   with the parent number baked into each list's --prefix. A legal answer that
   cannot say *which clause* it is quoting is close to useless, so the
   numbering is reconstructed from list position and re-inserted into the text.
2. Chunk boundaries. Each <h3> clause group is one chunk; oversized groups
   split at clause boundaries, never mid-clause.

Sections deep-link as ?section=N, so citations land on the right part of the
document rather than the top of a very long page.

Usage:
    python ingest_terms.py --repo ./oc --out corpus/terms.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

sys.path.insert(0, str(Path(__file__).parent))
from ingest_blog import node_to_text, preprocess, tidy
from ingest_guidelines import git_meta

TERMS_PATH = "frontend/app/src/components/landingpages/TermsContent.svelte"
TERMS_URL = "https://oc.app/terms"

TARGET_MAX_CHARS = 2000  # legal clauses are long; split above this
PREFIX_RE = re.compile(r"--prefix:\s*'([^']*)'")
SUFFIX_RE = re.compile(r"--suffix:\s*'([^']*)'")


def normalise_style_attrs(src: str) -> str:
    """style={"--prefix: 'A'"} -> style="--prefix: 'A'".

    The generic Svelte attribute rewrite in the blog preprocessor would turn
    this into an empty attribute and lose the numbering scheme with it.
    """
    return re.sub(r'style=\{"([^"]*)"\}', r'style="\1"', src)


def apply_clause_numbers(soup: BeautifulSoup) -> None:
    """Re-create the CSS-generated clause markers as real text.

    Each ``ul.custom_list`` renders ``--prefix`` + item counter + ``--suffix``
    before every direct child <li>, so "A1", "1.1)" and so on can be rebuilt
    from list position. The <li> is turned into a <p> afterwards: the generic
    renderer would otherwise prepend a bullet in front of the clause number.
    """
    for ul in soup.find_all("ul", class_="custom_list"):
        style = ul.get("style") or ""
        prefix_match, suffix_match = PREFIX_RE.search(style), SUFFIX_RE.search(style)
        prefix = prefix_match.group(1) if prefix_match else ""
        # The stylesheet default is ")"; an explicit empty --suffix overrides it.
        suffix = suffix_match.group(1) if suffix_match else ")"

        for counter, li in enumerate(ul.find_all("li", recursive=False), start=1):
            li.insert(0, NavigableString(f"{prefix}{counter}{suffix} "))
            li.name = "p"


def clause_groups(section: Tag) -> list[dict]:
    """Split a section into its <h3> clause groups, in document order."""
    groups: list[dict] = []
    current = {"heading": None, "nodes": []}

    def walk(node: Tag) -> None:
        nonlocal current
        for child in node.children:
            if isinstance(child, Tag) and child.name == "h3":
                if current["nodes"] or current["heading"]:
                    groups.append(current)
                current = {"heading": child.get_text(strip=True), "nodes": []}
            elif isinstance(child, Tag) and child.find("h3") is not None:
                # A wrapper (the outer numbered list) holding several groups.
                walk(child)
            else:
                current["nodes"].append(child)

    walk(section)
    if current["nodes"] or current["heading"]:
        groups.append(current)
    return groups


def split_long(text: str, limit: int = TARGET_MAX_CHARS) -> list[str]:
    """Split between clauses only — never inside one."""
    if len(text) <= limit:
        return [text]
    parts, current = [], ""
    for clause in text.split("\n"):
        if current and len(current) + len(clause) + 1 > limit:
            parts.append(current.strip())
            current = clause
        else:
            current = f"{current}\n{clause}" if current else clause
    if current.strip():
        parts.append(current.strip())
    return [part for part in parts if part]


SECTION_ANCHOR_RE = re.compile(r'<HashLinkTarget\b[^>]*\bid="(?P<id>\d+)"[^>]*>')


def split_sections(src: str) -> tuple[str, list[tuple[str, str]]]:
    """Split the raw source at section anchors: (preamble, [(id, block)]).

    Done on the text rather than the parsed tree because html.parser collapses
    the eleven sibling <div class="section"> elements into one — the custom
    component tags defeat its error recovery, and every clause then reports
    itself as belonging to Section A.
    """
    matches = list(SECTION_ANCHOR_RE.finditer(src))
    if not matches:
        return src, []
    preamble = src[: matches[0].start()]
    blocks = [
        (
            match.group("id"),
            src[match.start() : (matches[i + 1].start() if i + 1 < len(matches) else len(src))],
        )
        for i, match in enumerate(matches)
    ]
    return preamble, blocks


def parse_block(block: str) -> BeautifulSoup:
    soup = BeautifulSoup(preprocess(normalise_style_attrs(block)), "html.parser")
    apply_clause_numbers(soup)
    return soup


def build(repo: Path) -> list[dict]:
    src = (repo / TERMS_PATH).read_text()
    preamble_src, section_blocks = split_sections(src)

    provenance = git_meta(repo, TERMS_PATH)
    extracted_at = datetime.now(UTC).isoformat()
    chunks: list[dict] = []

    def emit(section_id: str, section_title: str, heading: str | None, text: str) -> None:
        breadcrumb = f"Terms of use > {section_title}"
        if heading:
            breadcrumb += f" > {heading}"
        for index, part in enumerate(split_long(text)):
            slug = re.sub(r"[^a-z0-9]+", "_", (heading or "intro").lower()).strip("_")[:40]
            suffix = "" if index == 0 else f".{index}"
            body = f"{breadcrumb}\n\n{part}"
            chunks.append(
                {
                    "id": f"terms:{section_id}:{slug}{suffix}",
                    "source_type": "terms",
                    "title": breadcrumb,
                    "text": body,
                    "url": f"{TERMS_URL}?section={section_id}",
                    "meta": {
                        "section_id": int(section_id),
                        "section_title": section_title,
                        "clause_group": heading,
                        "part": index,
                        "char_count": len(body),
                    },
                    "provenance": {**provenance, "extracted_at": extracted_at},
                    "content_hash": hashlib.sha256(body.encode()).hexdigest()[:16],
                }
            )

    # The preamble is the acceptance-by-use notice: worth keeping, it answers
    # "did I agree to anything by using OpenChat?".
    preamble_soup = parse_block(preamble_src)
    preamble = preamble_soup.find("div", class_="preamble")
    if preamble:
        text = tidy("".join(node_to_text(child) for child in preamble.children))
        if text:
            emit("0", "Preamble", None, text)

    for section_id, block in section_blocks:
        soup = parse_block(block)
        heading_tag = soup.find("h2")
        if not heading_tag:
            continue
        section_title = heading_tag.get_text(strip=True)
        heading_tag.extract()

        for group in clause_groups(soup):
            text = tidy("".join(node_to_text(node) for node in group["nodes"]))
            if len(text) < 40:  # structural leftovers, not clauses
                continue
            emit(section_id, section_title, group["heading"], text)

    if not chunks:
        raise SystemExit(
            f"no clauses parsed from {TERMS_PATH} — the markup has probably changed; "
            "fix the parser rather than shipping an empty source"
        )
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("corpus/terms.jsonl"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    chunks = build(args.repo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"{len(chunks)} chunks -> {args.out}")
    if args.verbose:
        for chunk in chunks:
            print(f"  {chunk['id']:52} {chunk['meta']['char_count']:>5}  {chunk['title'][:70]}")


if __name__ == "__main__":
    main()
