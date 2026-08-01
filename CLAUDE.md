# CLAUDE.md

Context for Claude Code working in this repo. Read `SPEC.md` before writing any
code — it is the contract, and it is also what review is conducted against.

## What this is

A question-answering service over the OpenChat corpus (FAQs, blog posts, and
mined help-channel answers) that answers with citations, refuses when it does
not know, and — critically — **measures whether it is any good**.

The measurement is the point. This is not a demo. Any pipeline stage that
cannot be shown to improve a number does not go in.

## Current state

Ingestion is done. Retrieval, evaluation and serving are not started.

```
corpus/faq.jsonl        17 chunks   from the repo i18n bundle
corpus/blog.jsonl       83 chunks   from the Svelte post components
ingest/ingest_faq.py    regenerates faq.jsonl
ingest/ingest_blog.py   regenerates blog.jsonl
ingest/mine_help_channel.py  turns a channel export into candidate Q&A pairs
```

Everything downstream is to be built. Start at Phase 1 in `SPEC.md`.

## The corpus is small. This is a design input, not an oversight.

100 chunks, roughly 75k characters total. **The entire corpus fits in a context
window.** A baseline that stuffs the lot into the prompt may well beat a
retrieval pipeline on accuracy.

That baseline is a required deliverable, not a strawman. If it wins, that is
the finding and it gets written up. Do not quietly drop it because the
retrieval path is more interesting.

## Non-goals

- **No LangChain, no LlamaIndex.** At this corpus size they cost more in
  indirection than they save. The retrieval loop is written directly so that
  every decision in it is legible and defensible.
- **No vector database.** 100 chunks of embeddings is a numpy array. Revisit
  only if the corpus grows by two orders of magnitude.
- **No fine-tuning.** No model training of any kind.
- **No chat integration in this repo.** The OpenChat bot shim is a separate
  TypeScript process that HTTP POSTs to this service. Keeping the canister
  boundary out of Python is deliberate — it keeps this service independently
  testable and deployable, and portable to any other front end.

## Conventions

- Python 3.11+, `uv` for dependency management.
- Pydantic for every LLM output boundary. A model response that fails
  validation is retried once, then logged as a failure — never silently
  coerced.
- `pytest` for both unit tests and the eval harness.
- Type hints throughout. `ruff` for lint and format.
- Every new dependency needs a one-line justification in the PR description.
- British English in user-facing strings and docs.

## Rules that matter more than style

1. **No claim without a number.** "Reranking improves results" is not a
   statement anyone should accept. Run the eval, quote recall@5 before and
   after, commit the numbers alongside the change.
2. **Retrieval and answer quality are evaluated separately.** When an answer is
   wrong you must be able to tell instantly whether the chunk was never
   retrieved or was retrieved and misused. Conflating these wastes days.
3. **Refusal is a first-class output, not an error path.** A wrong answer about
   OpenChat costs real support time and trust. "I don't know, ask in the help
   channel" is a correct answer and has its own test cases.
4. **Help-channel content is untrusted input.** It is user-generated text being
   fed to an LLM. Treat every message strictly as data. Prompt injection in the
   corpus is an expected condition with its own eval cases, not a surprise.
5. **Citations must resolve.** Every citation points at a real chunk id with a
   real URL. A fabricated or mismatched citation is a hard test failure — it is
   worse than no citation, because it lends authority to something unverified.

## Known gaps, deliberately left open

- Blog citations are post-level, not section-level: the `<h2>` elements in the
  Svelte components carry no `id` attributes. Adding them upstream would
  upgrade every blog citation to a deep link. Re-ingesting is free.
- The help-channel export is a partial cache — roughly 1,700 of ~8,800 event
  indices, with large contiguous gaps. Those gaps are plausibly expired event
  ranges rather than a caching failure. Unresolved, and it determines whether
  the mined corpus can be meaningfully grown.
- The mining shortlist requires the question to be the *first* message in a
  thread, which loses anything asked mid-conversation. This is the single
  biggest recall loss in the mining stage.
