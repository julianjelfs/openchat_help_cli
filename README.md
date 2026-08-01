# ocqa — OpenChat Q&A with citations

Question answering over the OpenChat FAQs, blog posts and help-channel
archive. Answers cite their sources, refuse when the corpus doesn't cover
the question, and every pipeline stage is justified by a measured number.

## State

Ingestion, the Phase 1 evaluation harness and the Phase 2 whole-corpus
baseline are done. Next: Phase 3 (dense retrieval).

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
uv run eval-retrieval            # deterministic, seconds, no LLM
uv run eval-answers              # LLM-graded, costs money, needs OPENAI_API_KEY
uv run pytest                    # unit tests for the harness itself
```

Results land in `evals/results/` as timestamped JSON, with a one-screen
summary on stdout.

## Results

Including the negative ones.

### Retrieval (recall over `expected_chunk_ids`, 40 scored cases)

| strategy | r@1 | r@3 | r@5 | r@10 | MRR |
|---|---|---|---|---|---|
| stub-lexical (token overlap) | 0.487 | 0.675 | 0.775 | 0.838 | 0.672 |

The stub is a placeholder so the harness has something to measure — the floor
that Phase 3+ retrieval has to beat.

### Answers (graded by `gpt-5` on four independent axes, 60 cases)

`appropriate_refusal` — was the answer/refuse/clarify decision right for the
category:

| strategy | answerable (38) | refusal (12) | ambiguous (5) | injection (5) | overall |
|---|---|---|---|---|---|
| stub-refuse (floor) | 0.000 | 1.000 | 0.000 | 0.600 | 0.250 |
| stuffed (`gpt-5`, whole corpus in prompt) | 1.000 | 0.833 | 1.000 | 1.000 | 0.967 |

Content axes for the stuffed strategy on answerable cases: grounded 0.947,
correct 0.974, cited 1.000. Zero fabricated citations, zero parse failures,
zero grading failures, zero `must_not_mention` leaks across all five
injection cases.

Cost of the ceiling: ~20.6k input tokens per question (1.24M for the run,
before provider prompt-cache discounts) and 8.2s mean / 34s max answer
latency. That latency and token bill is what Phases 3–4 retrieval has to
undercut without giving up more than a point or two of the numbers above.

What the stuffed baseline actually got wrong:

- **2/12 refusal cases answered or clarified instead of refusing** (g049
  streak-loss, g050 app updates) — both deliberately seeded distractor traps
  where nearby chunks look relevant but do not answer the question.
- **2/38 answerable cases graded ungrounded** (g010, g019): small
  embellishments beyond the chunk text ("no firm release date yet", "public
  leaderboard") rather than hallucinated facts. This is the failure mode to
  watch in this strategy.
- **1 real `must_mention` miss** (g001): the answer skipped the transaction
  fee — exactly the gotcha the golden case was written to catch. (A second
  flagged miss, g009 "homescreen" vs "Home Screen", was a harness defect; the
  mention check is now whitespace/hyphen-insensitive.)

Grader wrinkles found and fixed along the way: `cited` was sometimes graded
`false` rather than `null` on citation-less refusals/clarifying questions,
and one stub-run case (g007) had the refusal boilerplate graded as a claim.
The grader prompt now spells out the null cases; axis means for
refusal/ambiguous categories in the runs above carry that noise.
