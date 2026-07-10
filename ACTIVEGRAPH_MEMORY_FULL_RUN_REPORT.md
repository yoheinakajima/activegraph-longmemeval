# ActiveGraph Memory Pack Full LongMemEval Report

> Historical report: this documents the pre-v0.2 compiled-memory run at
> `0.878`. The typed projection, profile runtime, graph signal propagation,
> source-coverage proofs, durable fielded-vector cache, and retrieval artifacts
> described in the current adapter were implemented afterward. Keep this file
> as the baseline for the v0.2 comparison. See
> `ACTIVEGRAPH_MEMORY_V2_REPORT.md` for the complete follow-on experiment
> sequence and current pause-point analysis.

Report date: 2026-07-09

Primary repo: `yoheinakajima/activegraph-longmemeval`

Related pack repo: `yoheinakajima/activegraph-memory`

## Executive Result

The full `longmemeval_s_cleaned.json` run completed for `activegraph-memory-pack`
after adding resumable run state and a durable embedding cache.

Run artifact:

```text
runs/agmem-fullarch-full-20260709T022124Z__activegraph-memory-pack__s__full
```

Final score:

| Metric | Value |
|---|---:|
| Questions evaluated | 500 |
| Correct | 439 |
| Incorrect | 61 |
| Overall accuracy | 0.878 |
| Task-averaged accuracy | 0.8943 |
| Abstention accuracy | 0.9667 |
| Abstention questions | 30 |

This is a small gain over the strongest full `s` runs already present locally:

| System | Dataset | Questions | Overall | Task avg | Abstention |
|---|---|---:|---:|---:|---:|
| `rag-dense` turn | `longmemeval_s_cleaned.json` | 500 | 0.836 | 0.8600 | 0.9333 |
| `activegraph-det-embedding` | `longmemeval_s_cleaned.json` | 500 | 0.850 | 0.8772 | 0.9667 |
| `activegraph-sem-hybrid` | `longmemeval_s_cleaned.json` | 500 | 0.876 | 0.8835 | 0.9333 |
| `activegraph-memory-pack` | `longmemeval_s_cleaned.json` | 500 | 0.878 | 0.8943 | 0.9667 |

The result is directionally positive but not a large jump. The main remaining
failure modes are aggregate/counting questions, temporal filters, and
preference-recommendation questions where relevant profile evidence did not
enter the final context.

## Commits

LongMemEval repo:

```text
ce0cefc Add resumable full activegraph memory run
```

ActiveGraph Memory repo:

```text
55da1f7 Handle out-of-range temporal refs
```

Both commits were pushed to `main`.

## What Changed

### `activegraph-longmemeval`

Implemented durable/resumable benchmark execution:

- Added `src/activegraph_lme/embedding_cache.py`.
- Added a SQLite embedding cache at `.embedding_cache/embeddings.sqlite3`.
- Cache key: `(embedding_model, sha256(truncated_model_input))`.
- Stored vectors as normalized `float32` blobs.
- Added cache event rows for every newly stored vector.
- Integrated the cache into the shared `EmbeddingClient`.
- Reused the shared `EmbeddingClient` from `rag_dense` so dense baselines can use
  the same durable cache path.
- Added `AGLME_EMBEDDING_CACHE`:
  - unset: default `.embedding_cache/embeddings.sqlite3`
  - set to path: use that SQLite file
  - set to `off`/`false`/`0`: disable durable cache
- Added `AGLME_EMBED_BATCH_SIZE`, default `256`, capped at `2048`.

Implemented resumable run state:

- Added `--resume` to `activegraph_lme.cli run`.
- Added per-question durable records:
  - `query_records.jsonl`
  - `run_events.jsonl`
  - `run_state.json`
  - `manifest.partial.json`
- `hypotheses.jsonl` and `query_records.jsonl` are flushed and fsynced after each
  completed question.
- `run_state.json` and `manifest.partial.json` are rewritten after each completed
  question.
- On resume, completed question ids are loaded from `query_records.jsonl` and
  skipped.
- On resume, `hypotheses.jsonl` is repaired to exactly the completed prefix so
  duplicate hypotheses are not accumulated.
- `manifest.json` now records embedding-cache stats when the system uses
  embeddings.

Added test coverage:

- `scripts/property_tests.py` now includes a persistent embedding cache
  round-trip test.
- Existing activegraph-memory-pack contract test still passes.

Documentation:

- `README.md` now documents the full run result, resumable artifacts, and cache
  controls.
- `.gitignore` now ignores `.embedding_cache/`.

### `activegraph-memory`

Fixed a full-run crash in temporal normalization:

- Crash:

```text
OverflowError: date value out of range
```

- Source:

```text
activegraph_memory/temporal.py
```

- Cause:
  - The parser accepted numeric phrases like `10000 years ago`.
  - `anchor_date - timedelta(days=365 * 10000)` can underflow Python's valid
    `date` range.

- Fix:
  - Out-of-range relative temporal expressions now return a `TemporalRef` with
    `resolution_method="unresolved"` and metadata:

```json
{"reason": "relative_date_out_of_range"}
```

- Regression test:

```text
tests/test_compiler_retrieval.py::test_relative_ago_out_of_range_is_unresolved
```

## Run Configuration

Command used for the full run:

```bash
source <(textutil -convert txt -stdout .env.rtf)
LOG_LEVEL=WARNING AGLME_EMBED_BATCH_SIZE=256 /opt/homebrew/bin/python3.11 \
  -m activegraph_lme.cli run \
  --system activegraph-memory-pack \
  --dataset s \
  --run-id agmem-fullarch-full-20260709T022124Z \
  --resume \
  --require-authoritative-tokens
```

Eval command:

```bash
source <(textutil -convert txt -stdout .env.rtf)
LOG_LEVEL=WARNING /opt/homebrew/bin/python3.11 -m activegraph_lme.cli eval \
  --run-dir runs/agmem-fullarch-full-20260709T022124Z__activegraph-memory-pack__s__full
```

Relevant config values:

| Field | Value |
|---|---|
| Dataset | `data/longmemeval_s_cleaned.json` |
| System | `activegraph-memory-pack` |
| Reader requested | `claude-sonnet-4-5` |
| Reader resolved | `claude-sonnet-4-5-20250929` |
| Reader temperature | `0.0` |
| Reader max tokens | `1024` |
| Judge | `gpt-4o-2024-08-06` via upstream short name `gpt-4o` |
| Embedding model | `text-embedding-3-small` |
| ActiveGraph token budget | `2500` |
| Token source | `tiktoken` |
| Extraction seed | `A-v2` |

Token/run stats from `manifest.json`:

| Stat | Value |
|---|---:|
| Mean context tokens | 2483.9 |
| Mean prompt tokens | 2908.3 |
| Mean completion tokens | 90.9 |
| Truncated contexts | 500 / 500 |
| Manifest wall clock seconds | 7312.716 |

Every context was marked `truncated`. This does not mean every answer was bad;
it means the activegraph memory retrieval pool produced more candidate evidence
than the fixed `2500` token budget could carry for every query.

## Artifact Inventory

Directory:

```text
runs/agmem-fullarch-full-20260709T022124Z__activegraph-memory-pack__s__full
```

Files:

| File | Purpose | Size bytes |
|---|---|---:|
| `hypotheses.jsonl` | Reader answers submitted to judge | 205766 |
| `hypotheses.jsonl.eval-results-gpt-4o` | Upstream judge per-question labels | 238387 |
| `scores.json` | Final aggregate scores | 601 |
| `manifest.json` | Final provenance manifest | 119619 |
| `manifest.partial.json` | Last partial manifest from resumable run | 119587 |
| `query_records.jsonl` | Durable per-question records for resume | 90532 |
| `run_events.jsonl` | Append-only run event stream | 272966 |
| `run_state.json` | Latest resume pointer/state | 9901 |
| `eval.log` | Upstream eval log | 327214 |

Run event counts:

| Event type | Count |
|---|---:|
| `run.started` | 1 |
| `run.resumed` | 2 |
| `query.started` | 502 |
| `query.completed` | 500 |
| `run.failed` | 1 |
| `run.completed` | 1 |

There are two more `query.started` events than completed queries because:

1. The first execution was deliberately interrupted after verifying resume at
   21 completed questions.
2. The second execution crashed at 220 completed questions on the temporal
   overflow bug, then resumed after the fix.

## Embedding Cache Stats

From `manifest.json`:

| Stat | Value |
|---|---:|
| Embed requests through `EmbeddingClient` | 479310 |
| In-memory hits | 50751 |
| Persistent hits | 58597 |
| Misses | 369962 |
| API batches | 1946 |
| API texts | 369962 |
| Persistent cache rows at manifest time | 714884 |
| Persistent cache events at manifest time | 714884 |
| Persistent stores in this process | 369345 |
| Persistent store conflicts | 617 |

Important interpretation:

- The full run was not a clean cold cache. A one-question probe and earlier smoke
  work had already populated some embeddings.
- The full run still stored 369345 new vectors during the final process.
- The cache now makes future reruns materially cheaper and restartable.
- The cache is local and gitignored. It was not committed.

## Score Breakdown

| Question type | Accuracy | Correct | Wrong | N |
|---|---:|---:|---:|---:|
| `single-session-user` | 0.9571 | 67 | 3 | 70 |
| `single-session-preference` | 0.8333 | 25 | 5 | 30 |
| `single-session-assistant` | 0.9821 | 55 | 1 | 56 |
| `multi-session` | 0.7895 | 105 | 28 | 133 |
| `temporal-reasoning` | 0.8421 | 112 | 21 | 133 |
| `knowledge-update` | 0.9615 | 75 | 3 | 78 |

Largest contributors to error count:

1. `multi-session`: 28 wrong
2. `temporal-reasoning`: 21 wrong
3. `single-session-preference`: 5 wrong

The high single-session lookup scores show the pack can surface local evidence.
The weaker multi-session and temporal scores show the current implementation is
not yet doing enough structured aggregation, date filtering, and preference
state retrieval.

## Failure Pattern Analysis

The following heuristic tags were assigned from the failed question text. Tags
can overlap, so totals exceed 61.

| Failure pattern | Count among wrong answers |
|---|---:|
| Count/total/aggregate wording | 31 |
| Temporal filter/order wording | 25 |
| Lookup or other | 13 |
| Preference/recommendation wording | 3 |

Concrete implication:

- The largest gap is not semantic recall in general.
- The largest gap is operation execution over recalled evidence:
  - count all matching events
  - sum all matching quantities
  - include/exclude by date range
  - order events chronologically
  - reconcile updated facts

The current system still mostly retrieves evidence and asks the reader to do the
operation inside a compact context. That works on many examples, but it fails
when the correct answer requires complete coverage across several sessions.

## Representative Failures

These are examples from `hypotheses.jsonl.eval-results-gpt-4o`. Hypotheses are
truncated here for readability.

| Type | QID | Pattern | Question | Reference answer | Model hypothesis excerpt |
|---|---|---|---|---|---|
| `knowledge-update` | `0f05491a` | count/total | How many stars do I need to reach the gold level on my Starbucks Rewards app? | 120 | Answered 125 stars. |
| `knowledge-update` | `41698283` | lookup/update | What type of camera lens did I purchase most recently? | 70-200mm zoom lens | Answered 50mm prime lens. |
| `knowledge-update` | `69fee5aa` | count/total | How many pre-1920 American coins do I have in my collection? | 38 | Answered 37. |
| `multi-session` | `0a995998` | count/total | How many items of clothing do I need to pick up or return from a store? | 3 | Answered 2, missing one item. |
| `multi-session` | `3a704032` | count + temporal | How many plants did I acquire in the last month? | 3 | Answered only 1 plant. |
| `multi-session` | `gpt4_d84a3211` | sum + temporal | How much total money have I spent on bike-related expenses since the start of the year? | $185 | Answered $65. |
| `multi-session` | `gpt4_7fce9456` | count + exclusion | How many properties did I view before making an offer on the townhouse in the Brookside neighborhood? | 4 | Answered at least 3. |
| `single-session-preference` | `0edc2aef` | preference recommendation | Can you suggest a hotel for my upcoming trip to Miami? | Prefer great views and unique features such as rooftop pool or hot tub balcony. | Claimed insufficient preference history. |
| `single-session-preference` | `32260d93` | preference recommendation | Can you recommend a show or movie for me to watch tonight? | Prefer stand-up comedy specials on Netflix. | Claimed insufficient preference history. |
| `single-session-user` | `51a45a95` | lookup | Where did I redeem a $5 coupon on coffee creamer? | Target | Claimed store was not specified. |
| `single-session-assistant` | `1903aded` | list position | What was the 7th work-from-home job in the list you provided? | Transcriptionist | Answered online survey taker. |
| `temporal-reasoning` | `gpt4_7f6b06db` | chronological order | What is the order of the three trips I took in the past three months, from earliest to latest? | Muir Woods, Big Sur, Eastern Sierra. | Identified only Eastern Sierra. |
| `temporal-reasoning` | `gpt4_1916e0ea` | date difference | How many days passed between cancelling FarmFresh and Instacart shopping? | 54 days, or 55 including last day. | Claimed insufficient information. |
| `temporal-reasoning` | `gpt4_7abb270c` | chronological order | What is the order of the six museums I visited from earliest to latest? | Six-museum ordered list. | Produced an incomplete/incorrect ordering. |

## Failed Question IDs By Type

`knowledge-update`:

```text
0f05491a, 41698283, 69fee5aa
```

`multi-session`:

```text
0a995998, 3a704032, gpt4_d84a3211, dd2973ad, c4a1ceb8,
gpt4_a56e767c, gpt4_2f8be40d, gpt4_7fce9456, gpt4_5501fe77,
2ce6a0f2, gpt4_31ff4165, a9f6b44c, d851d5ba, gpt4_ab202e7f,
gpt4_731e37d7, bf659f65, a11281a2, 51c32626, 681a1674,
a1cc6108, 9ee3ecd6, 92a0aa75, 60159905, 73d42213, 09ba9854,
37f165cf, 21d02d0d, 09ba9854_abs
```

`single-session-assistant`:

```text
1903aded
```

`single-session-preference`:

```text
0edc2aef, 32260d93, 09d032c9, d6233ab6, 1c0ddc50
```

`single-session-user`:

```text
51a45a95, ec81a493, 8a137a7f
```

`temporal-reasoning`:

```text
gpt4_7f6b06db, gpt4_1916e0ea, gpt4_7abb270c, gpt4_45189cb4,
370a8ff4, gpt4_d6585ce8, 6e984301, gpt4_f420262c,
gpt4_21adecb5, eac54adc, 71017277, gpt4_f420262d,
gpt4_e414231f, gpt4_4929293b, eac54add, 0bc8ad93,
a3838d2b, 982b5123, gpt4_9a159967, d01c6aa8, gpt4_2f56ae70
```

## Why The Score Is Not Higher

### 1. Fixed 2500-token budget loses coverage on aggregate questions

Every question was marked `truncated`. Multi-session aggregate questions often
require all supporting sessions. Missing one session changes the answer, even if
the retrieved context is mostly relevant.

Examples:

- `0a995998`: expected 3 clothing pickup/return items, answered 2.
- `3a704032`: expected 3 plants acquired in last month, answered 1.
- `gpt4_d84a3211`: expected $185 bike expenses, answered $65.

This is a retrieval coverage failure first, not primarily a reader failure.

### 2. The pack retrieves evidence but does not execute structured operations

The compiled memory index has claims, temporal refs, quantity claims, source
turns, and evidence bundles. The final answer still depends on the reader to
perform counting, summing, set membership, and chronological ordering over the
selected context.

For questions like "How many", "How much total", "in the last month", and
"from earliest to latest", the memory layer should compute candidate operations
before the reader sees the final context.

### 3. Temporal normalization exists but is not yet a full temporal query engine

The temporal code can normalize simple references and prevent crashes, but the
retrieval layer does not yet guarantee complete coverage of all events within a
relative date window.

Examples:

- `gpt4_1916e0ea`: needs date difference between two events.
- `gpt4_7f6b06db`: needs complete trip ordering over the past three months.
- `gpt4_7abb270c`: needs ordering of six museums.

### 4. Preference memory is not first-class enough

Preference questions are only 30 examples, but they hurt task average. Failures
show the system sometimes responds as if no personalization evidence exists.

Examples:

- `0edc2aef`: missed Miami hotel preferences.
- `32260d93`: missed Netflix stand-up comedy preference.
- `09d032c9`: missed prior portable power bank context.

The pack needs explicit preference/profile claim typing and retrieval rules, not
only generic semantic claim retrieval.

### 5. Smoke result was optimistic

The 50-question smoke run scored:

```text
overall accuracy:       0.94
task-averaged accuracy: 0.9634
abstention accuracy:    1.0
```

The full 500 scored:

```text
overall accuracy:       0.878
task-averaged accuracy: 0.8943
abstention accuracy:    0.9667
```

The smoke run did not expose enough of the difficult multi-session aggregate and
temporal coverage cases. It was useful as a build gate, not as an estimate of
full-set performance.

## Reliability Outcome

The resumable architecture was tested by real interruption and real failure.

Observed sequence:

1. Started the full run.
2. Interrupted intentionally after 21 completed questions to verify resume.
3. Resumed from 21 completed questions.
4. Crashed at 220 completed questions on temporal date underflow.
5. Fixed `activegraph-memory`.
6. Resumed from 220 completed questions.
7. Completed 500/500.
8. Ran official eval.

The important result: neither interruption nor crash required recomputing the
completed prefix. The persistent embedding cache also preserved expensive
embedding work.

## Tests Run

ActiveGraph Memory:

```bash
/opt/homebrew/bin/python3.11 -m pytest
```

Result:

```text
28 passed
```

LongMemEval property tests:

```bash
make tests PY=/opt/homebrew/bin/python3.11
```

Result:

```text
all property tests passed (skips do not count as failures)
```

The embedding sub-variant is skipped in the offline property suite when
`OPENAI_API_KEY` is not present in that process, which is expected for the
offline gate.

## Recommended Next Work

### P0: Add retrieval recall diagnostics

Add a sidecar that compares retrieved source turn/session ids against
`answer_session_ids` before the reader runs.

Required output per question:

- selected turn ids
- selected session ids
- selected claim ids
- answer session ids
- recall hit/miss
- number of answer sessions covered
- whether miss happened before reader generation

This will separate retrieval failures from reader reasoning failures.

### P1: Add aggregate-aware query planning

For queries with count/sum/order words:

- increase evidence budget dynamically
- diversify retrieval by session
- force inclusion of more top sessions instead of repeated near-duplicate claim
  evidence
- expose a structured scratch table to the reader:
  - event
  - quantity
  - date
  - source session
  - confidence

This directly targets the 31 failed count/total questions.

### P1: Add symbolic reducers for quantities and counts

Use compiled `QuantityClaim` and temporal refs to compute candidate answers for:

- sums
- counts
- date differences
- earliest/latest ordering
- membership within date windows

The reader should verify or verbalize a computed result, not discover the whole
operation from raw context.

### P1: Make preference memory first-class

Add claim types or metadata for:

- preference
- constraint
- disliked option
- recommendation rationale
- stable user profile fact

Preference/recommendation queries should route to this lane before generic
semantic retrieval.

### P2: Temporal index and temporal operators

Build an index over:

- session date
- claim observed date
- normalized temporal refs
- explicit event dates
- relative date windows resolved against `question_date`

Then support operations:

- `within(window)`
- `before(event)`
- `after(event)`
- `latest(property)`
- `earliest_to_latest(events)`
- `days_between(event_a, event_b)`

This targets the 25 failed questions with temporal-filter wording.

### P2: Run ablations on full 500

Run these cells on the full `s` dataset:

1. Current pack with dynamic budget.
2. Current pack with session-diversified retrieval.
3. Current pack without memory-claim headers.
4. Current pack with aggregate-aware planning.
5. Current pack with symbolic reducers.

Keep the same reader, judge, model snapshots, and token accounting.

## Bottom Line

The new architecture is worth continuing. It now beats the strongest local full
`s` baselines by a small margin and has the operational machinery needed for
larger experiments: resumable runs, durable embedding cache, event stream, and
crash recovery.

The next accuracy gains should not come from generic more-of-the-same semantic
retrieval. The failure distribution points to missing structured operations:
complete evidence coverage, counting, summing, temporal windows, ordering, and
preference-profile routing.
