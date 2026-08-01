"""Human review gate for mined help-channel candidates.

    uv run review-candidates

Help-channel text is untrusted user content; nothing enters the index without
a human decision (SPEC.md data contract, CLAUDE.md rule 4). This tool makes
that decision explicit and durable: approvals land in ``corpus/help.jsonl``
with ``status: "approved"`` and a review timestamp, rejections are remembered
in ``corpus/help_rejected.txt`` so a re-mine never resurfaces them, and
already-decided candidates are skipped, so the review is resumable.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


def load_decided(out_path: Path, rejects_path: Path) -> tuple[set[str], set[str]]:
    approved = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                approved.add(json.loads(line)["id"])
    rejected = set()
    if rejects_path.exists():
        rejected = {
            line.split()[0] for line in rejects_path.read_text().splitlines() if line.strip()
        }
    return approved, rejected


def approve(candidate: dict, out_path: Path) -> None:
    candidate = dict(candidate)
    candidate["status"] = "approved"
    candidate["provenance"] = {
        **candidate.get("provenance", {}),
        "reviewed_at": datetime.now(UTC).isoformat(),
    }
    with out_path.open("a") as fh:
        fh.write(json.dumps(candidate, ensure_ascii=False) + "\n")


def reject(candidate_id: str, rejects_path: Path) -> None:
    with rejects_path.open("a") as fh:
        fh.write(f"{candidate_id} {datetime.now(UTC).isoformat()}\n")


def show(candidate: dict, position: int, total: int) -> None:
    meta = candidate["meta"]
    stale = " STALE — re-verify before trusting" if meta.get("stale") else ""
    print("=" * 78)
    print(
        f"[{position}/{total}] {candidate['id']}  "
        f"confidence {meta['confidence']:.2f}  answered {meta['answered_at'][:10]}{stale}"
    )
    print(f"url: {candidate['url']}")
    print(f"\nQ: {meta['question']}\n")
    print(f"A: {meta['answer']}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Review mined help-channel candidates")
    parser.add_argument("--candidates", type=Path, default=Path("corpus/help_candidates.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("corpus/help.jsonl"))
    parser.add_argument("--rejects", type=Path, default=Path("corpus/help_rejected.txt"))
    parser.add_argument("--approve", default=None, help="comma-separated ids (non-interactive)")
    parser.add_argument("--reject", default=None, help="comma-separated ids (non-interactive)")
    args = parser.parse_args()

    candidates = [
        json.loads(line)
        for line in args.candidates.read_text().splitlines()
        if line.strip()
    ]
    by_id = {candidate["id"]: candidate for candidate in candidates}
    already_approved, already_rejected = load_decided(args.out, args.rejects)
    decided = already_approved | already_rejected

    if args.approve or args.reject:
        approve_ids = [i.strip() for i in (args.approve or "").split(",") if i.strip()]
        reject_ids = [i.strip() for i in (args.reject or "").split(",") if i.strip()]
        unknown = [i for i in approve_ids + reject_ids if i not in by_id]
        if unknown:
            sys.exit(f"unknown candidate ids: {unknown}")
        overlap = set(approve_ids) & set(reject_ids)
        if overlap:
            sys.exit(f"ids in both --approve and --reject: {sorted(overlap)}")
        for cid in approve_ids:
            if cid in decided:
                print(f"{cid}: already decided, skipping")
                continue
            approve(by_id[cid], args.out)
            print(f"{cid}: approved")
        for cid in reject_ids:
            if cid in decided:
                print(f"{cid}: already decided, skipping")
                continue
            reject(cid, args.rejects)
            print(f"{cid}: rejected")
        return

    pending = [candidate for candidate in candidates if candidate["id"] not in decided]
    if not pending:
        print("Nothing to review: every candidate is already decided.")
        return
    print(f"{len(pending)} candidates to review ({len(decided)} already decided).")
    print("Keys: [a]pprove  [r]eject  [s]kip  [q]uit\n")

    for position, candidate in enumerate(pending, start=1):
        show(candidate, position, len(pending))
        while True:
            choice = input("approve/reject/skip/quit [a/r/s/q]: ").strip().lower()
            if choice in {"a", "r", "s", "q"}:
                break
        if choice == "q":
            break
        if choice == "a":
            approve(candidate, args.out)
        elif choice == "r":
            reject(candidate["id"], args.rejects)

    approved_now, rejected_now = load_decided(args.out, args.rejects)
    print(
        f"\napproved: {len(approved_now)}, rejected: {len(rejected_now)}, "
        f"undecided: {len(by_id) - len(approved_now | rejected_now)}"
    )
    print(f"approved chunks live in {args.out} and are indexed on the next corpus load.")


if __name__ == "__main__":
    main()
