# SPEC — OpenChat Q&A service

Implementation contract. Phases are ordered by dependency and by what de-risks
the project soonest. **Do not skip ahead to Phase 3.** The evaluation harness
exists before the thing it evaluates, deliberately: build retrieval first and
you will tune it on vibes for a week.

Each phase has acceptance criteria. A phase is done when they pass, not when
the code runs.

---

## Data contract

Every chunk, from every source, is one JSON object per line:

```json
{
  "id": "faq:security",
  "source_type": "faq | blog | help_channel",
  "title": "Are my messages secure?",
  "text": "<what gets embedded>",
  "url": "https://oc.app/faq?q=security",
  "meta": { "...source-specific..." },
  "provenance": { "...where it came from..." },
  "content_hash": "a1b2c3d4e5f60718"
}
```

`id` is stable across re-ingests and is what citations reference. `text`
already carries its own context — the FAQ question, or the blog breadcrumb —
so a chunk retrieved in isolation still says what it is about.

Help-channel chunks additionally carry `status`, which must be `approved`
before indexing. `pending` chunks are never retrievable.

---

## Phase 1 — Evaluation harness

Build this first, against the corpus that already exists.

### 1.1 Golden set

`evals/golden.jsonl`, 40–60 cases minimum. Four categories, all required:

| Category | What it tests | Minimum |
|---|---|---|
| `answerable` | Question the corpus genuinely answers | 25 |
| `refusal` | Plausible OpenChat question the corpus does *not* answer | 8 |
| `ambiguous` | Underspecified; a clarifying question is the right response | 4 |
| `injection` | Text attempting to redirect the model's behaviour | 3 |

Case format:

```json
{
  "id": "g001",
  "category": "answerable",
  "question": "How do I get my CHAT tokens to an exchange?",
  "expected_chunk_ids": ["faq:wallet", "faq:send_tokens"],
  "must_mention": ["wallet", "transaction fee"],
  "must_not_mention": [],
  "notes": "Fee point is the bit people actually get caught by"
}
```

`expected_chunk_ids` is the ground truth for retrieval scoring, so it must be
filled in by hand against the actual corpus. This is tedious and it is the most
valuable hour of the project.

Seed the `answerable` cases from real questions in the help-channel export
where a human gave a good answer — those are real user phrasing, which is
worth far more than questions invented by whoever is holding the keyboard.

### 1.2 Retrieval eval

Deterministic, no LLM, runs in seconds. `uv run eval-retrieval`.

- recall@1, @3, @5, @10 against `expected_chunk_ids`
- MRR
- per-category and per-`source_type` breakdown
- output to `evals/results/<timestamp>-retrieval.json` **and** a one-screen
  summary to stdout

### 1.3 Answer eval

LLM-graded, expensive, run deliberately. `uv run eval-answers`.

Grade each answer on four independent axes. Do not collapse them into one
score; a single number hides exactly the failure you need to see.

- **grounded** — every claim traceable to a cited chunk
- **correct** — factually right per the cited material
- **cited** — citations present, resolvable, and actually relevant
- **appropriate_refusal** — refused when it should have, answered when it should have

For `refusal` cases, answering confidently is the failure mode being measured.
For `injection` cases, any deviation from normal answering behaviour is a
failure — including a refusal that quotes the injected instruction back.

Acceptance: both commands run against the current corpus and produce committed
baseline numbers, with a stub retriever if necessary.

---

## Phase 2 — Whole-corpus baseline

The control. `strategy=stuffed`.

Concatenate every approved chunk into the prompt, ask the question, require the
same structured output as every other strategy. Measure it with the Phase 1
harness. Commit the numbers.

This is roughly 20k tokens per call — slow and not cheap, but it establishes
the ceiling that retrieval has to justify itself against.

Acceptance: baseline scores committed to `evals/results/`. If this beats
everything in Phases 3–4, say so plainly in the README.

---

## Phase 3 — Dense retrieval

`strategy=dense`. Deliberately unsophisticated.

- Embed every chunk. Embeddings cached to disk, keyed on `content_hash`, so
  re-ingesting does not mean re-embedding.
- Store as a single numpy array. Brute-force cosine similarity.
- Top-k into the prompt.

Acceptance: recall@5 reported; answer eval run; both compared against Phase 2
in the same table.

---

## Phase 4 — Hybrid and reranking

Each step is a separate change with its own measured delta. If a step does not
move the numbers, it does not ship — and that negative result gets written down.

### 4.1 BM25 + reciprocal rank fusion — `strategy=hybrid`

Justified rather than reflexive here: the corpus is dense with product jargon
(canister, chit, CHAT, ICP, neuron, diamond membership) and dense embeddings
are weak on exact terms and acronyms. Expect this to matter most on
`buychat`-style questions.

### 4.2 Cross-encoder reranking — `strategy=hybrid+rerank`

Rerank the top ~30 down to the top ~5. Report the latency cost alongside the
accuracy gain; if it doubles response time for one point of recall, that is a
finding worth stating.

Acceptance: a results table comparing all four strategies on identical golden
cases, committed to the repo.

---

## Phase 5 — Service

FastAPI. `POST /ask`.

```
Request:  { "question": str, "strategy": str = "hybrid+rerank", "max_chunks": int = 5 }
Response: { "answer": str,
            "citations": [ { "chunk_id": str, "url": str, "title": str,
                             "source_type": str, "published": str | null } ],
            "refused": bool,
            "confidence": float,
            "strategy": str,
            "latency_ms": int }
```

Requirements:

- Answer generation returns a Pydantic-validated object; one retry on parse
  failure, then a refusal rather than an unvalidated answer.
- Every returned `chunk_id` must exist in the index. Validate before responding
  and fail the request loudly if not — a fabricated citation must never reach a
  user.
- `refused: true` returns an empty citation list and a pointer to the help
  channel.
- Structured logging: question, strategy, retrieved ids, refusal, latency.
  These logs are the next eval set.
- `/health` returns corpus size and index build time.

Acceptance: service runs; eval harness can target the HTTP endpoint as well as
the library directly; injection cases pass against the live service.

---

## Phase 6 — Out of scope here

The OpenChat bot shim: a thin TypeScript process using the bot SDK that
receives a command, POSTs to `/ask`, and posts the answer with citation links.
No retrieval logic, no prompts, no domain knowledge. Separate repo.

Two things to settle before building it: verify the message permalink format
against a real client link, and decide whether mined help-channel answers are
attributed by link only (current assumption) or not surfaced at all.

---

## Definition of done

- All four strategies implemented and measured on the same golden set.
- A results table in the README with real numbers, including the ones that
  didn't help.
- Refusal and injection cases passing.
- A written statement of which pipeline stages earned their place and which
  were dropped, with the numbers that justified each decision.

That last item is the actual deliverable. The service is the artefact; the
measured reasoning about it is the thing worth showing anyone.
