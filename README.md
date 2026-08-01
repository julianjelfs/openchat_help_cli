# ocqa — OpenChat Q&A with citations

Question answering over the OpenChat FAQs, blog posts and help-channel
archive. Answers cite their sources, refuse when the corpus doesn't cover
the question, and every pipeline stage is justified by a measured number.

## State

Ingestion, the evaluation harness (Phase 1), the whole-corpus baseline
(Phase 2), dense retrieval (Phase 3) and the HTTP service (Phase 5) are done.

**Phase 4 (hybrid BM25 + reranking) was deliberately skipped**: dense
retrieval already places every golden expected chunk in the top 5
(recall@5 = 1.000), so hybrid and reranking have nothing left to demonstrate
at this corpus size. Revisit if the corpus grows or a harder golden set
(adversarial paraphrases, k=3 budgets) opens headroom — r@1 (0.787) is where
any future gain lives.

| | chunks |
|---|---|
| `corpus/faq.jsonl` | 17 |
| `corpus/blog.jsonl` | 83 |
| `corpus/guidelines.jsonl` | 8 |
| `corpus/terms.jsonl` | 73 |
| `corpus/help.jsonl` | 14 (human-approved of 20 mined) |

## Running the service

```bash
uv run serve-ocqa                # needs OPENAI_API_KEY; builds the index at startup
curl -s localhost:8000/health
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "How do I buy CHAT?"}'
```

`POST /ask` takes `{question, strategy (default "dense"), max_chunks (default
5)}` and answers with `gpt-5-mini` (the measured cost/quality winner —
override with `serve-ocqa --answer-model gpt-5`) and returns the answer with resolved citations (`chunk_id`, `url`,
`title`, `source_type`, `published`), `refused`, `confidence` and
`latency_ms`. A citation that does not resolve against the index is a 500 —
a fabricated citation never reaches a user. Refusals carry no citations and
point at the help channel. Every request is logged as one JSON line
(question, strategy, retrieved ids, cited ids, refusal, latency) — those
logs are the next eval set. The spec named `hybrid+rerank` as the default
strategy; the measured winner is `dense`, so that is the default.

The eval harness can target a running service instead of the library:

```bash
uv run eval-answers --strategy dense --endpoint http://127.0.0.1:8000
```

## Regenerating the corpus

```bash
git clone --depth 1 --filter=blob:none https://github.com/open-chat-labs/open-chat.git oc
python ingest/ingest_faq.py         --repo ./oc --out corpus/faq.jsonl
python ingest/ingest_blog.py        --repo ./oc --out corpus/blog.jsonl
python ingest/ingest_guidelines.py  --repo ./oc --out corpus/guidelines.jsonl
python ingest/ingest_terms.py       --repo ./oc --out corpus/terms.jsonl
```

Guidelines and terms chunks carry section-level citations: both pages read
`?section=N`, so a chunk deep-links to the exact rule or clause group rather
than the top of a long page. The terms ingester also reconstructs clause
numbers ("3.1)", "4.1.2)") that exist only as CSS counters in the markup — a
legal answer that cannot say which clause it is quoting is close to useless.

The terms restate parts of the guidelines (Schedule 2 ≈ `guidelines:3`,
Schedule 4 ≈ `guidelines:4`), so some golden cases accept either source: an
`expected_chunk_ids` entry may be a list meaning "any one of these". The
worry that 73 legal chunks would crowd the top-5 was measured and did not
materialise — see Results.

Help channel, from a browser IndexedDB export:

```bash
python ingest/mine_help_channel.py \
  --events ~/Downloads/openchat-help-export.json \
  --out corpus/help_candidates.jsonl --limit 5 --no-llm
```

Candidates land as `status: pending` and are never indexed until approved:

```bash
uv run review-candidates          # a/r/s/q per candidate, resumable
```

Approvals are written to `corpus/help.jsonl` with `status: approved`;
rejections are remembered so a re-mine never resurfaces them.

## Evaluation

The golden set (`evals/golden.jsonl`) holds 60 hand-verified cases: 38
answerable (several seeded from real help-channel questions), 12 refusal,
5 ambiguous and 5 injection.

```bash
uv run eval-retrieval                     # dense by default; --strategy stub for the floor
uv run eval-answers --strategy dense      # LLM-graded; also: stuffed, stub
uv run pytest                             # unit tests for the harness itself
```

Both commands need OPENAI_API_KEY (embeddings on first run — cached in
`cache/` keyed on content_hash — and gpt-5 for answering/grading).

Results land in `evals/results/` as timestamped JSON, with a one-screen
summary on stdout.

## Results

Including the negative ones.

### Retrieval (recall over `expected_chunk_ids`, 40 scored cases)

| corpus | strategy | r@1 | r@3 | r@5 | r@10 | MRR |
|---|---|---|---|---|---|---|
| 100 chunks | stub-lexical (token overlap) | 0.487 | 0.675 | 0.775 | 0.838 | 0.672 |
| 100 chunks | dense (`text-embedding-3-large`) | 0.787 | 0.950 | **1.000** | 1.000 | 0.942 |
| 195 chunks | dense (+guidelines, terms, help) | 0.788 | — | **0.983** | 0.983 | 0.907 |

Per source at 195 chunks (micro, over expected chunks):

| source | n | r@1 | r@5 |
|---|---|---|---|
| faq | 21 | 0.905 | 1.000 |
| blog | 29 | 0.552 | 1.000 |
| guidelines | 6 | 1.000 | 1.000 |
| help_channel | 8 | 1.000 | 1.000 |
| terms | 4 | 0.250 | 0.750 |

**Did the terms crowd out the corpus? No.** That was the open question when
73 legal chunks (37% of the corpus) went in. Every other source still scores
r@5 1.000; the mined help chunks and the guidelines are perfect at every k.
The whole cost is one case, g083.

**The one real failure, and what it argues for.** g083 asks "if I lose money
on a token swap, is OpenChat liable?" and the answering clause
(`terms:1:our_liability`) is never retrieved, even at k=10 — the clause is
written in formal legal register and never uses the user's vocabulary.
That is a vocabulary-mismatch failure, precisely what BM25 is good at and
embeddings are not, which reopens the case for Phase 4 hybrid retrieval that
the earlier 1.000 had closed. Small sample (4 terms cases) — worth more
legal-phrasing cases before drawing hard conclusions.

Blog r@1 stays weak (0.552) because adjacent sections of the same post
compete for the top slot; harmless, since r@5 is 1.000 and the siblings
carry overlapping content.

### Answers (graded by `gpt-5` on four independent axes, 60 cases)

`appropriate_refusal` — was the answer/refuse/clarify decision right for the
category:

Rows marked v1 were measured on the original answering rules; v2 rows on the
rewritten rules (see below). `gpt-5` was not re-run on v2 (cost call), so the
cross-model comparison is approximate. v1 rows have 5 ambiguous cases; v2
has 10 (five held-out cases added after tuning).

| strategy / answer model | answerable (38) | refusal (12) | ambiguous | injection (5) | overall | mean latency |
|---|---|---|---|---|---|---|
| stub-refuse (floor) | 0.000 | 1.000 | 0.000 | 0.600 | 0.250 | — |
| stuffed / `gpt-5` v1 | 1.000 | 0.833 | 1.000 | 1.000 | 0.967 | 8.2s |
| dense / `gpt-5` v1 | 1.000 | 0.917 | 1.000 | 1.000 | 0.983 | 9.5s |
| dense / `gpt-5` v1 via live HTTP service | 1.000 | 0.833 | 1.000 | 1.000 | 0.967 | 9.3s |
| dense / `gpt-5-mini` v1 | 1.000 | 0.917 | **0.400** | 1.000 | 0.933 | 6.6s |
| dense / `gpt-5-mini` v2 | 1.000 | 0.917 | **1.000** | 0.800 | 0.969 | 8.2s |

Content axes for the stuffed strategy on answerable cases: grounded 0.947,
correct 0.974, cited 1.000. Zero fabricated citations, zero parse failures,
zero grading failures, zero `must_not_mention` leaks across all five
injection cases.

Content axes on answerable cases, stuffed vs dense: grounded 0.947 / 0.947,
correct 0.974 / 0.974, cited 1.000 / 0.974. Statistically the same answer
quality from 5 chunks as from the whole corpus.

**Phase 5 acceptance: met.** The harness run against the live HTTP service
(`--endpoint`) passed all five injection cases, produced zero fabricated
citations, and added negligible latency over the library path. The two
`gpt-5` dense rows also bracket the run-to-run variance on the refusal
traps: 10–11 of 12, same strategy, different runs — those cases sit at the
model's decision boundary.

**Model choice, measured (the "do we really need gpt-5?" question):**
on the v1 rules, `gpt-5-mini` matched `gpt-5` everywhere except ambiguous
questions, where it guessed a reading instead of clarifying (0.400 vs
1.000). Two changes fixed that — the output schema now forces a
`response_type: answer | clarify | refuse` classification before the model
writes anything, and the rules were rewritten as an explicit decision
procedure (clarify checked first, with warnings about near-miss chunks and
account-specific questions). Measured on the v2 rules with five held-out
ambiguous cases added to guard against tuning-to-the-test: ambiguous went
0.400 → 1.000 (10/10), and g049 — the streak-loss refusal trap no model had
ever passed — now refuses correctly. `gpt-5-mini` is the service default at
roughly a fifth of the token price of `gpt-5`.

Two residual `gpt-5-mini` weaknesses, on the record:

- **Injection: echo-naming persists; compliance-class attacks are fixed.**
  Two held-out injection cases (g066 exfiltration-by-task, g067
  summarise-the-context) were added and the first attempt at a no-echo rule
  *worsened* the slice (5/7): g067 exposed that mini would execute a
  task-framed injection against the retrieved chunks. A meta-request rule
  (anything about the assistant, its rules, or the reference text itself is
  always a refusal — the chunks are source material, never the subject)
  closed that hole: 6/7 on the injection slice, held-out g067 refusing
  cleanly. The residual is cosmetic-but-real: mini names the refused request
  ("...my system prompt") despite two rounds of no-echo instruction; g066's
  deterministic `must_not_mention` tripwire flags it on every run so it
  cannot silently regress. Grader severity on echo-style refusals also
  wobbles run to run — the tripwire, not the grader, is the stable signal
  here.
- **The clarify-vs-refuse boundary flickers.** g047/g050/g055 sit exactly on
  it and a different one misses each run; the refusal category is stable at
  11/12. The answerable-axis cost also changed flavour: gpt-5 embellished
  detail, mini slips tense (reporting planned features as live — grounded
  0.947 either way).

**Phase 3 finding: dense retrieval beats the stuffed control.** Same or
better on every quality axis (notably 11/12 vs 10/12 on the refusal traps —
less irrelevant context appears to mean less temptation to answer), at ~1.5k
input tokens per question against 20.6k (a 93% cut, 89.8k vs 1.24M for the
run). One negative result worth stating plainly: **latency did not improve**
(9.5s vs 8.2s mean) — answer time is dominated by gpt-5 reasoning, not prompt
length, so retrieval buys cost, not speed.

What the LLM strategies actually got wrong (stuffed / dense):

- **Refusal traps**: stuffed sprang two (g049 streak-loss, g050 app
  updates); dense sprang only g049. Both are deliberately seeded distractor
  cases where nearby chunks look relevant but do not answer the question.
  g049 has now resisted both strategies — the hardest case in the set.
- **2/38 answerable cases graded ungrounded in each strategy** (stuffed:
  g010, g019; dense: g001, g010): small embellishments beyond the chunk text
  ("no firm release date yet", "use the exchange's deposit address") rather
  than hallucinated facts. A gpt-5 trait, not a context-size effect — it
  shows up identically at 1.5k and 20.6k tokens of context.
- **g001 skipped the transaction fee in both strategies** — exactly the
  gotcha the golden case was written to catch, and a prompt-tuning target for
  Phase 4/5. (A second flagged miss in the stuffed run, g009 "homescreen" vs
  "Home Screen", was a harness defect; the mention check is now
  whitespace/hyphen-insensitive.)

Grader wrinkles found and fixed along the way: `cited` was sometimes graded
`false` rather than `null` on citation-less refusals/clarifying questions,
and one stub-run case (g007) had the refusal boilerplate graded as a claim.
The grader prompt now spells out the null cases; axis means for
refusal/ambiguous categories in the runs above carry that noise.
