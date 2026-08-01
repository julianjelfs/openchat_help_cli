# ocqa — OpenChat Q&A with citations

Question answering over the OpenChat FAQs, blog posts and help-channel
archive. Answers cite their sources, refuse when the corpus doesn't cover
the question, and every pipeline stage is justified by a measured number.

## State

Ingestion, the evaluation harness (Phase 1), the whole-corpus baseline
(Phase 2) and dense retrieval (Phase 3) are done. Next: Phase 4 (hybrid +
reranking) — which now has to justify itself against dense's recall@5 of
1.000 on this golden set.

| | chunks |
|---|---|
| `corpus/faq.jsonl` | 17 |
| `corpus/blog.jsonl` | 83 |
| help channel | mining pipeline built, not yet run |

## Regenerating the corpus

```bash
git clone --depth 1 --filter=blob:none https://github.com/open-chat-labs/open-chat.git oc
python ingest/ingest_faq.py  --repo ./oc --out corpus/faq.jsonl
python ingest/ingest_blog.py --repo ./oc --out corpus/blog.jsonl
```

Help channel, from a browser IndexedDB export:

```bash
python ingest/mine_help_channel.py \
  --events ~/Downloads/openchat-help-export.json \
  --out corpus/help_candidates.jsonl --limit 5 --no-llm
```

Candidates land as `status: pending` and are never indexed until approved.

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

| strategy | r@1 | r@3 | r@5 | r@10 | MRR |
|---|---|---|---|---|---|
| stub-lexical (token overlap) | 0.487 | 0.675 | 0.775 | 0.838 | 0.672 |
| dense (`text-embedding-3-large`, cosine) | 0.787 | 0.950 | **1.000** | 1.000 | 0.942 |

The stub is a placeholder floor. Dense retrieval puts every expected chunk in
the top 5, so a 5-chunk answer prompt never misses the material it needs.
Dense r@1 is weaker on blog chunks (0.633 vs 0.850 for FAQ) because adjacent
sections of the same post compete for the top slot — mostly harmless, since
the siblings carry overlapping content and r@3 recovers to 0.933.

### Answers (graded by `gpt-5` on four independent axes, 60 cases)

`appropriate_refusal` — was the answer/refuse/clarify decision right for the
category:

| strategy | answerable (38) | refusal (12) | ambiguous (5) | injection (5) | overall |
|---|---|---|---|---|---|
| stub-refuse (floor) | 0.000 | 1.000 | 0.000 | 0.600 | 0.250 |
| stuffed (`gpt-5`, whole corpus in prompt) | 1.000 | 0.833 | 1.000 | 1.000 | 0.967 |
| dense (`gpt-5`, top-5 retrieved chunks) | 1.000 | 0.917 | 1.000 | 1.000 | 0.983 |

Content axes for the stuffed strategy on answerable cases: grounded 0.947,
correct 0.974, cited 1.000. Zero fabricated citations, zero parse failures,
zero grading failures, zero `must_not_mention` leaks across all five
injection cases.

Content axes on answerable cases, stuffed vs dense: grounded 0.947 / 0.947,
correct 0.974 / 0.974, cited 1.000 / 0.974. Statistically the same answer
quality from 5 chunks as from the whole corpus.

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
