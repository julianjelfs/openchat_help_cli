# ocqa — OpenChat Q&A with citations

Question answering over the OpenChat FAQs, blog posts and help-channel
archive. Answers cite their sources, refuse when the corpus doesn't cover
the question, and every pipeline stage is justified by a measured number.

## State

Ingestion done, everything else to build. Start with `SPEC.md` Phase 1.

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

## Results

To be filled in from `evals/results/`. Including the negative ones.
