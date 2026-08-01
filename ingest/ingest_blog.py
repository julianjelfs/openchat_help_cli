#!/usr/bin/env python3
"""
Extract the OpenChat blog posts into the flat corpus format.

The posts are hand-written Svelte components rather than markdown, but the
markup is clean and semantic: <section> wrappers with <h2>/<h3> headings and
<p> bodies. Those section boundaries are the chunk boundaries, and the headings
give free breadcrumbs.

    frontend/app/src/components/landingpages/blog/posts.ts   -> registry
    frontend/app/src/components/landingpages/blog/*.svelte   -> post bodies

Usage:
    python ingest_blog.py --repo ./oc --out corpus/blog.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

BLOG_DIR = "frontend/app/src/components/landingpages/blog"
POSTS_TS = f"{BLOG_DIR}/posts.ts"
BLOG_BASE_URL = "https://oc.app/blog"

TARGET_MAX_CHARS = 1500  # split above this, at paragraph boundaries
TARGET_MIN_CHARS = 250   # merge adjacent sections below this

SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
SVELTE_BLOCK_RE = re.compile(r"\{[#/:@][^}]*\}")
CLASS_DIRECTIVE_RE = re.compile(r"\sclass:[\w-]+=\{[^}]*\}")
EVENT_HANDLER_RE = re.compile(r"\son\w+=\{[^}]*\}")
WS_RE = re.compile(r"[ \t\n]+")


@dataclass
class Chunk:
    id: str
    source_type: str
    title: str
    text: str
    url: str
    meta: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    content_hash: str = ""

    def finalise(self) -> Chunk:
        self.content_hash = hashlib.sha256(self.text.encode()).hexdigest()[:16]
        return self


# --- registry ---------------------------------------------------------------

def js_date(year: int, month: int, day: int) -> date:
    """Replicate `new Date(y, m, d)`.

    The month argument is zero indexed, and out-of-range days roll forward
    rather than erroring: one post is dated new Date(2024, 1, 31), i.e. the
    31st of February, which the site renders as 2 March 2024. Reproduce that
    rather than correcting it, so extracted dates match what users see.
    """
    year += month // 12
    return date(year, month % 12 + 1, 1) + timedelta(days=day - 1)


def parse_registry(repo: Path) -> list[dict]:
    """Read slug/title/author/date/component out of posts.ts.

    NB: these are JS Date literals, where the month argument is ZERO indexed.
    new Date(2026, 6, 26) is 26 July 2026, not June. Getting this wrong shifts
    every post by a month and quietly poisons any recency weighting.
    """
    src = (repo / POSTS_TS).read_text()

    imports = dict(re.findall(r'import\s+(\w+)\s+from\s+"\./([\w]+)\.svelte"', src))

    entries = []
    pattern = re.compile(
        r'slug:\s*"(?P<slug>[^"]+)",\s*'
        r'title:\s*"(?P<title>(?:[^"\\]|\\.)*)",\s*'
        r'author:\s*"(?P<author>[^"]*)",\s*'
        r'date:\s*new Date\((?P<y>\d+),\s*(?P<m>\d+),\s*(?P<d>\d+)\),\s*'
        r'component:\s*(?P<component>\w+),',
        re.DOTALL,
    )
    for m in pattern.finditer(src):
        component = m.group("component")
        if component not in imports:
            print(f"  ! no import found for component {component}")
            continue
        entries.append({
            "slug": m.group("slug"),
            "title": m.group("title").replace('\\"', '"'),
            "author": m.group("author"),
            "date": js_date(int(m.group("y")), int(m.group("m")), int(m.group("d"))),
            "file": f"{imports[component]}.svelte",
        })
    return entries


# --- svelte -> text ---------------------------------------------------------

def preprocess(src: str) -> str:
    """Strip the bits an HTML parser has no business seeing, and rewrite the
    custom components into plain tags it does understand."""
    src = SCRIPT_STYLE_RE.sub("", src)

    # <Markdown text="..."> carries literal content (one post embeds an ASCII
    # diagram this way). Pull the attribute out as a <pre> block.
    def markdown_component(m: re.Match) -> str:
        body = m.group("text").replace("&quot;", '"')
        return f"<pre>{body}</pre>"

    src = re.sub(
        r'<Markdown\b[^>]*?text="(?P<text>.*?)"\s*/?>(?:</Markdown>)?',
        markdown_component, src, flags=re.DOTALL,
    )

    # <BlogScreenshot caption="..." .../> -> keep the caption, drop the image.
    src = re.sub(
        r'<BlogScreenshot\b[^>]*?caption="(?P<caption>[^"]*)"[^>]*?/?>(?:</BlogScreenshot>)?',
        lambda m: f'<p class="figure">Screenshot: {m.group("caption")}</p>',
        src, flags=re.DOTALL,
    )

    # <ExternalLink href="x">y</ExternalLink> and <Link path="x">y</Link> -> <a>
    src = re.sub(r"<ExternalLink\b([^>]*)>", r"<a\1>", src)
    src = src.replace("</ExternalLink>", "</a>")
    src = re.sub(r"<Link\b([^>]*?)path=", r"<a\1href=", src)
    src = re.sub(r"<Link\b([^>]*)>", r"<a\1>", src)
    src = src.replace("</Link>", "</a>")

    src = CLASS_DIRECTIVE_RE.sub("", src)
    src = EVENT_HANDLER_RE.sub("", src)
    src = SVELTE_BLOCK_RE.sub("", src)
    src = re.sub(r"=\{([^}]*)\}", r'="\1"', src)  # attr={x} -> attr="x"
    return src


def node_to_text(node: Tag | NavigableString) -> str:
    if isinstance(node, NavigableString):
        return WS_RE.sub(" ", str(node))
    if node.name in {"script", "style"}:
        return ""
    if node.name == "pre":
        return f"\n```\n{node.get_text()}\n```\n"
    if node.name == "br":
        return "\n"
    if node.name == "img":
        alt = (node.get("alt") or "").strip()
        return f"\n[image: {alt}]\n" if alt else ""

    inner = "".join(node_to_text(c) for c in node.children)

    if node.name == "a":
        href = (node.get("href") or "").strip()
        label = inner.strip()
        if not href or href.startswith(("{", "$")):
            return label
        if href.startswith("/"):
            href = f"https://oc.app{href}"
        return f"[{label}]({href})" if label else href
    if node.name in {"strong", "b"}:
        return f"**{inner.strip()}**"
    if node.name in {"em", "i"}:
        return f"_{inner.strip()}_"
    if node.name == "code":
        return f"`{inner.strip()}`"
    if node.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        return f"\n\n{'#' * int(node.name[1])} {inner.strip()}\n\n"
    if node.name == "li":
        return f"\n- {inner.strip()}"
    if node.name in {"p", "ul", "ol", "div", "blockquote", "table", "tr"}:
        return f"\n{inner.strip()}\n"
    if node.name in {"td", "th"}:
        return f"{inner.strip()} | "
    return inner


def tidy(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_sections(src: str) -> list[dict]:
    """One entry per <section>, carrying its heading and body text.

    Falls back to splitting on <h2> for any post that does not use sections.
    """
    soup = BeautifulSoup(preprocess(src), "html.parser")
    sections = soup.find_all("section")

    if not sections:
        blocks, current = [], {"heading": None, "nodes": []}
        for node in soup.children:
            if isinstance(node, Tag) and node.name == "h2":
                if current["nodes"]:
                    blocks.append(current)
                current = {"heading": node.get_text(strip=True), "nodes": []}
            else:
                current["nodes"].append(node)
        if current["nodes"]:
            blocks.append(current)
        return [
            {
                "heading": b["heading"],
                "subheading": None,
                "text": tidy("".join(node_to_text(n) for n in b["nodes"])),
            }
            for b in blocks
        ]

    out = []
    for section in sections:
        heading_tag = section.find(["h2", "h3"])
        heading = heading_tag.get_text(strip=True) if heading_tag else None
        if heading_tag:
            heading_tag.extract()

        # h3s are subheadings within a section. Split on them: they are better
        # chunk boundaries than an arbitrary character count, and they give the
        # breadcrumb another level of specificity.
        blocks = [{"subheading": None, "nodes": []}]
        for node in section.children:
            if isinstance(node, Tag) and node.name == "h3":
                blocks.append({"subheading": node.get_text(strip=True), "nodes": []})
            else:
                blocks[-1]["nodes"].append(node)

        for block in blocks:
            text = tidy("".join(node_to_text(n) for n in block["nodes"]))
            if text or block["subheading"]:
                out.append({
                    "heading": heading,
                    "subheading": block["subheading"],
                    "text": text,
                })
    return out


# --- chunking ---------------------------------------------------------------

def split_long(text: str, limit: int) -> list[str]:
    """Split on blank lines, never mid-paragraph."""
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

    # A single paragraph can still exceed the limit (long unbroken prose, a
    # table). Fall back to sentence boundaries rather than shipping a chunk
    # that swamps the context of everything retrieved alongside it.
    final: list[str] = []
    for part in parts:
        if len(part) <= limit:
            final.append(part)
            continue
        buf = ""
        for sentence in re.split(r"(?<=[.!?])\s+", part):
            if buf and len(buf) + len(sentence) + 1 > limit:
                final.append(buf.strip())
                buf = sentence
            else:
                buf = f"{buf} {sentence}" if buf else sentence
        if buf.strip():
            final.append(buf.strip())
    return final


def chunk_post(post: dict, sections: list[dict]) -> list[Chunk]:
    # Merge runs of very short sections so we don't index one-line fragments.
    merged: list[dict] = []
    for section in sections:
        if not section["text"]:
            continue
        if merged and len(merged[-1]["text"]) < TARGET_MIN_CHARS:
            prev = merged[-1]
            prev["heading"] = prev["heading"] or section["heading"]
            prev["subheading"] = prev["subheading"] or section["subheading"]
            prev["text"] = f"{prev['text']}\n\n{section['text']}".strip()
        else:
            merged.append(dict(section))

    url = f"{BLOG_BASE_URL}/{post['slug']}"
    chunks: list[Chunk] = []

    for section_no, section in enumerate(merged):
        for part_no, body in enumerate(split_long(section["text"], TARGET_MAX_CHARS)):
            breadcrumb = " > ".join(
                x for x in [post["title"], section["heading"], section["subheading"]] if x
            )
            suffix = f".{part_no}" if part_no else ""
            chunks.append(
                Chunk(
                    id=f"blog:{post['slug']}:{section_no}{suffix}",
                    source_type="blog",
                    title=breadcrumb,
                    # Breadcrumb is prepended so a chunk retrieved in isolation
                    # still says which post and section it came from.
                    text=f"{breadcrumb}\n\n{body}",
                    url=url,
                    meta={
                        "slug": post["slug"],
                        "post_title": post["title"],
                        "section_heading": section["heading"],
                        "section_subheading": section["subheading"],
                        "section_index": section_no,
                        "part": part_no,
                        "author": post["author"],
                        "published": post["date"].isoformat(),
                        "char_count": len(body),
                    },
                    provenance={
                        "repo": "open-chat-labs/open-chat",
                        "path": f"{BLOG_DIR}/{post['file']}",
                        "extracted_at": datetime.now(UTC).isoformat(),
                    },
                ).finalise()
            )
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("oc"))
    ap.add_argument("--out", type=Path, default=Path("corpus/blog.jsonl"))
    args = ap.parse_args()

    posts = parse_registry(args.repo)
    print(f"registry: {len(posts)} posts")

    all_chunks: list[Chunk] = []
    for post in sorted(posts, key=lambda p: p["date"], reverse=True):
        path = args.repo / BLOG_DIR / post["file"]
        if not path.exists():
            print(f"  ! missing {path}")
            continue
        sections = parse_sections(path.read_text())
        chunks = chunk_post(post, sections)
        all_chunks.extend(chunks)
        print(f"  {post['date']}  {post['slug']:<22} {len(sections):>2} sections -> {len(chunks):>2} chunks")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for chunk in all_chunks:
            fh.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    sizes = [len(c.text) for c in all_chunks]
    print(f"\nwrote {len(all_chunks)} chunks -> {args.out}")
    print(f"chars: min {min(sizes)}  max {max(sizes)}  mean {sum(sizes)//len(sizes)}")


if __name__ == "__main__":
    main()
