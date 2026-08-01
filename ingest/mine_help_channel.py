#!/usr/bin/env python3
"""
Mine candidate question/answer pairs out of an OpenChat help channel.

This consumes a JSON export you produce yourself with the OpenChat TS/Rust
agent -- a list of EventWrapper<Message> from chatEvents(). Nothing here talks
to a canister; keeping the network boundary in TypeScript and the pipeline in
Python is the whole point.

Expected export shape (map the SDK types down to this before running):

    [
      {
        "index": 41233,                     # EventWrapper.index
        "timestamp": 1739812345678,         # ms since epoch
        "messageIndex": 8811,
        "senderId": "abcde-...",
        "text": "how do I get my CHAT out of OpenChat?",
        "threadRootMessageIndex": null,     # set on thread replies
        "repliesToEventIndex": null,        # ReplyContext.eventIndex
        "reactions": {"thumbsup": 3},
        "isBot": false,
        "deleted": false
      }
    ]

Pipeline:
    1. clean       -- drop bots, commands, deleted, trivia
    2. thread      -- rebuild conversations from thread/reply edges, then time
    3. shortlist   -- cheap heuristics, so the LLM only sees plausible threads
    4. extract     -- schema-validated LLM pass, one call per thread
    5. review      -- write candidates as status="pending"; nothing is indexed
                      until a human approves it

Usage:
    python mine_help_channel.py --events export.json --out corpus/help_candidates.jsonl
    python mine_help_channel.py --events export.json --out ... --no-llm   # dry run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Iterable

from pydantic import BaseModel, Field, ValidationError

# --- config -----------------------------------------------------------------

COMMUNITY_ID = "dgegb-daaaa-aaaar-arlhq-cai"
CHANNEL_ID = "3798400021"
# Verify this against a real message link from the client before trusting it.
PERMALINK = "https://oc.app/community/{community}/channel/{channel}/{message_index}"

THREAD_GAP = timedelta(minutes=45)  # fallback grouping when no reply edge
MIN_ANSWER_CHARS = 60
MAX_THREAD_MESSAGES = 40
STALE_AFTER_DAYS = 270  # older pairs get flagged for re-verification

QUESTION_HINTS = re.compile(
    r"(\?|^how\b|^why\b|^what\b|^where\b|^when\b|^can i\b|^is it\b|^does\b|^anyone know\b)",
    re.IGNORECASE,
)
NOISE = re.compile(r"^\s*(/\w+|[\W_]{1,4}|gm|ty|thanks|thx|lol|\+1)\s*$", re.IGNORECASE)


# --- LLM output contract ----------------------------------------------------

class ExtractedPair(BaseModel):
    """What the model is allowed to give back. Anything else is a parse failure."""

    is_qa: bool = Field(description="Does this thread contain a clear question and a clear answer?")
    question: str = Field(default="", description="Rewritten as a standalone question, no chat context")
    answer: str = Field(default="", description="Rewritten as a standalone answer, no chat context")
    answer_event_index: int | None = Field(default=None, description="Event index of the message the answer came from")
    self_contained: bool = Field(default=False, description="False if it leans on images, links or earlier context not present")
    is_product_question: bool = Field(default=False, description="False for price talk, chit-chat, moderation, off-topic")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = Field(default="", description="One line: why accepted or rejected")


SYSTEM_PROMPT = """You extract support knowledge from OpenChat help channel threads.

Return a single JSON object matching the schema. No prose, no markdown fences.

Reject the thread (is_qa: false) if ANY of the following hold:
- No clear question, or no clear answer to it
- The answer is speculation, a guess, or contradicted later in the thread
- The answer only makes sense with context outside the excerpt (screenshots,
  "as I said above", "do what he suggested")
- It is about token price, chit-chat, moderation, or a one-off account issue
- The answer is a promise about future work rather than how the product works

Rewrite accepted pairs so they stand alone: no usernames, no "hi", no thread
references. The answer must be usable by someone who has never seen the thread.
Prefer the answer that the original asker confirmed or reacted positively to.
Set confidence below 0.7 whenever you are unsure. Being conservative is correct;
a wrong answer with a citation attached is worse than no answer."""


# --- steps ------------------------------------------------------------------

def load_events(path: str) -> list[dict]:
    raw = json.load(open(path))
    # The browser export wraps the array in a header object; a hand-rolled SDK
    # exporter will more likely hand you the bare list. Accept either.
    events = raw["events"] if isinstance(raw, dict) else raw
    kept = [
        e for e in events
        if not e.get("deleted")
        and not e.get("isBot")
        and (e.get("text") or "").strip()
        and not NOISE.match(e["text"])
        and not e["text"].lstrip().startswith("/")
    ]
    print(f"  {len(events)} events -> {len(kept)} after cleaning")
    return sorted(kept, key=lambda e: e["index"])


def build_threads(events: list[dict]) -> list[list[dict]]:
    """Explicit thread and reply edges first; time-window clustering only as a
    fallback for the messages that carry neither."""
    by_event_index = {e["index"]: e for e in events}
    by_message_index = {e["messageIndex"]: e for e in events if e.get("messageIndex") is not None}
    groups: dict[int, list[dict]] = defaultdict(list)
    loose: list[dict] = []

    def root_of(event: dict, depth: int = 0) -> int | None:
        if depth > 10:
            return None
        if (tri := event.get("threadRootMessageIndex")) is not None:
            root = by_message_index.get(tri)
            return root["index"] if root else None
        if (rei := event.get("repliesToEventIndex")) is not None:
            parent = by_event_index.get(rei)
            return root_of(parent, depth + 1) or (parent["index"] if parent else None)
        return None

    for event in events:
        root = root_of(event)
        if root is not None:
            groups[root].append(event)
        else:
            loose.append(event)

    for root_index in list(groups):
        if (root := by_event_index.get(root_index)) is not None:
            groups[root_index].insert(0, root)
            loose = [e for e in loose if e["index"] != root_index]

    # Fallback: consecutive loose messages within THREAD_GAP become one thread.
    current: list[dict] = []
    for event in loose:
        ts = datetime.fromtimestamp(event["timestamp"] / 1000, timezone.utc)
        if current:
            prev = datetime.fromtimestamp(current[-1]["timestamp"] / 1000, timezone.utc)
            if ts - prev > THREAD_GAP:
                groups[current[0]["index"]] = list(current)
                current = []
        current.append(event)
    if current:
        groups[current[0]["index"]] = list(current)

    threads = [sorted(g, key=lambda e: e["index"])[:MAX_THREAD_MESSAGES] for g in groups.values()]
    print(f"  {len(threads)} threads reconstructed")
    return threads


def shortlist(threads: Iterable[list[dict]]) -> list[list[dict]]:
    """Cheap filters before spending tokens. An LLM call per thread over a
    multi-year channel gets expensive fast, and most threads are not Q&A."""
    out = []
    for thread in threads:
        if len(thread) < 2:
            continue
        if not QUESTION_HINTS.search(thread[0]["text"]):
            continue
        replies = thread[1:]
        if not any(len(r["text"]) >= MIN_ANSWER_CHARS for r in replies):
            continue
        if len({m["senderId"] for m in thread}) < 2:
            continue
        out.append(thread)
    print(f"  {len(out)} threads shortlisted for extraction")
    return out


def render(thread: list[dict]) -> str:
    """Pseudonymise senders. The model never needs real user IDs, and neither
    does the eventual citation."""
    aliases: dict[str, str] = {}
    lines = []
    for m in thread:
        alias = aliases.setdefault(m["senderId"], f"user{len(aliases) + 1}")
        reactions = f"  [reactions: {m['reactions']}]" if m.get("reactions") else ""
        lines.append(f"[event {m['index']}] {alias}: {m['text']}{reactions}")
    return "\n".join(lines)


def extract(thread: list[dict], client) -> ExtractedPair | None:
    """One schema-validated call, with a single retry on a parse failure."""
    body = render(thread)
    messages = [{"role": "user", "content": body}]

    for attempt in range(2):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        raw = response.content[0].text.strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw).strip()
        try:
            return ExtractedPair.model_validate_json(raw)
        except ValidationError as err:
            if attempt == 0:
                messages += [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"That failed validation:\n{err}\nReturn valid JSON only."},
                ]
            else:
                print(f"  ! validation failed twice on thread at event {thread[0]['index']}")
                return None


def to_candidate(thread: list[dict], pair: ExtractedPair) -> dict:
    by_event = {m["index"]: m for m in thread}
    source = by_event.get(pair.answer_event_index) or thread[-1]
    answered_at = datetime.fromtimestamp(source["timestamp"] / 1000, timezone.utc)
    text = f"{pair.question}\n\n{pair.answer}"

    return {
        "id": f"help:{thread[0]['index']}",
        "source_type": "help_channel",
        "status": "pending",  # never indexed until a human flips this
        "title": pair.question,
        "text": text,
        "url": PERMALINK.format(
            community=COMMUNITY_ID, channel=CHANNEL_ID,
            message_index=source.get("messageIndex", source["index"]),
        ),
        "meta": {
            "question": pair.question,
            "answer": pair.answer,
            "confidence": pair.confidence,
            "self_contained": pair.self_contained,
            "reason": pair.reason,
            "answered_at": answered_at.isoformat(),
            "stale": datetime.now(timezone.utc) - answered_at > timedelta(days=STALE_AFTER_DAYS),
            "thread_size": len(thread),
        },
        "provenance": {
            "community_id": COMMUNITY_ID,
            "channel_id": CHANNEL_ID,
            "root_event_index": thread[0]["index"],
            "answer_event_index": source["index"],
            # Deliberately no sender IDs: cite the message, not the person.
            "extracted_at": datetime.now(timezone.utc).isoformat(),
        },
        "content_hash": hashlib.sha256(text.encode()).hexdigest()[:16],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--out", default="corpus/help_candidates.jsonl")
    ap.add_argument("--min-confidence", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=None, help="cap threads sent to the LLM")
    ap.add_argument("--no-llm", action="store_true", help="stop after shortlisting")
    args = ap.parse_args()

    threads = shortlist(build_threads(load_events(args.events)))
    if args.limit:
        threads = threads[: args.limit]
    if args.no_llm:
        print("--no-llm set, stopping. Sample shortlisted thread:\n")
        if threads:
            print(render(threads[0])[:1500])
        return

    from anthropic import Anthropic

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    accepted, rejected = [], 0

    for i, thread in enumerate(threads, 1):
        pair = extract(thread, client)
        if pair is None:
            continue
        keep = (
            pair.is_qa and pair.self_contained and pair.is_product_question
            and pair.confidence >= args.min_confidence
        )
        if keep:
            accepted.append(to_candidate(thread, pair))
        else:
            rejected += 1
        if i % 25 == 0:
            print(f"  {i}/{len(threads)} processed, {len(accepted)} kept")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        for candidate in accepted:
            fh.write(json.dumps(candidate, ensure_ascii=False) + "\n")

    stale = sum(1 for c in accepted if c["meta"]["stale"])
    print(f"\nkept {len(accepted)}, rejected {rejected} -> {args.out}")
    print(f"{stale} flagged stale (older than {STALE_AFTER_DAYS} days)")
    print("All candidates are status=pending. Review before indexing.")


if __name__ == "__main__":
    main()
