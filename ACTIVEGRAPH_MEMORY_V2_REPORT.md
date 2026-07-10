# ActiveGraph Memory v2 LongMemEval Report

Report date: 2026-07-10

Benchmark repository: `yoheinakajima/activegraph-longmemeval`

Memory package repository: `yoheinakajima/activegraph-memory`

Status: superseded by `ACTIVEGRAPH_MEMORY_V3_REPORT.md` for the 0.3.0 smoke and
full-500 results. This file remains the historical v2 pause-point record.

This report is the current pause-point record for the ActiveGraph Memory work.
It complements `ACTIVEGRAPH_MEMORY_FULL_RUN_REPORT.md`, which remains the
historical record for the earlier `0.878` run. It does not replace or reinterpret
that result.

## Executive Summary

The work produced a substantially more capable memory package, but the newest
full LongMemEval score regressed from the historical best.

| Run | Correct | Wrong | Overall | Task average | Abstention |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical compiled-memory full run | 439 | 61 | 0.878 | 0.8943 | 0.9667 |
| Latest typed v2 proof run | 412 | 88 | 0.824 | 0.8515 | 0.9333 |

The final v2 artifact reports 88 wrong answers. A previously discussed value of
70 misses is not the final 500-question score and should not be compared with
the historical 61 without naming the subset or metric. This report treats each
run's `scores.json` as authoritative for QA accuracy.

The regression is not explained by a collapse in raw retrieval recall. On the
470 non-abstention questions, v2 retrieved at least one exact gold source turn
for 90.43% of questions, slightly above the earlier semantic-hybrid system's
90.00%. The largest measured change was downstream: v2 had 63 incorrect reader
answers despite an exact gold turn being present, versus 37 for semantic hybrid.

The strongest current interpretation is:

1. Typed retrieval and graph propagation improved or preserved broad source
   recall.
2. Deterministic compiled candidates were sometimes semantically wrong despite
   satisfying structural proof fields.
3. The reader prompt told the reader to trust a `Verified candidate`, causing
   wrong compiled answers to anchor final answers.
4. The 10,000-token context carried more evidence but also more distractors;
   487 of 500 contexts still reached the configured budget.
5. The benchmark did not attach an optional LLM reasoner, so the `max_quality`
   profile exercised deterministic classification, expansion, typed execution,
   and packaging only.

The post-run code now calls these outputs `Proof-complete candidate` and
`Incomplete candidate`, and explicitly says proof completion certifies evidence
field presence rather than answer correctness. That correction is tested but
has not yet been measured in a new smoke or full run.

## Frozen Evaluation Boundary

All compared ActiveGraph Memory runs used the same external answer and judge
boundary unless a run manifest says otherwise:

| Component | Setting |
| --- | --- |
| Dataset | `data/longmemeval_s_cleaned.json` |
| Questions | 500 |
| Reader requested | `claude-sonnet-4-5` |
| Reader resolved | `claude-sonnet-4-5-20250929` |
| Reader temperature | 0.0 |
| Reader max output | 1,024 tokens |
| Reader tools/web | disabled |
| Judge | `gpt-4o-2024-08-06` through upstream `gpt-4o` |
| Judge temperature | 0.0 |
| Embeddings | `text-embedding-3-small` |
| Extraction input | role-aware `seed-A-v2` cache |
| Token accounting | `tiktoken` plus authoritative API usage |

The fixed reader is intentional. Changes in QA score therefore measure the
joint effect of retrieval, compiled evidence, context ordering, and reader use
of that context. Alternate reader models are useful product experiments but
must be reported as separate cells.

## What Was Built In `activegraph-memory`

The package evolved from a claim retriever into a typed, profile-driven memory
runtime.

### Compiled memory plane

- canonical entities and aliases
- event mentions with actual/planned/hypothetical/recommended modality
- canonical event deduplication
- event time separated from observation time
- versioned state histories and supersession
- scoped preference and professional-profile evidence
- structured quantities and measures
- normalized temporal references
- position-preserving assistant list items
- source and claim provenance on every compiled row

### Query and execution plane

- multi-operator query IR
- lookup, count, sum, maximum, latest, current, previous, temporal order,
  date delta, ordinal, negative-existence, and recommendation operators
- explicit operator-specific proof requirements
- category/action-bounded aggregate scans
- item cardinality rather than row cardinality
- canonical-event deduplication before aggregate execution
- source-coverage checks
- role-aware retrieval for user versus assistant history
- time windows resolved against the question date
- typed candidate answers plus source-grounded evidence rows

### Retrieval plane

- lexical candidates over source turns and claims
- fielded embeddings over claims, turns, entities, events, states, and
  preferences
- reciprocal-rank fusion across lexical, caller, and embedding signals
- entity score propagation to neighboring claims, turns, events, and states
- targeted query variants and multiple retrieval rounds
- session-diverse source packaging
- persistent corpus vectors through SQLite or ActiveGraph objects

Embeddings are therefore used as graph signals, not as a standalone top-k list.
They generate and rank candidates; typed execution and source checks decide
whether an operator has the fields it requires.

### Optional reasoning and profiles

The runtime exposes four independent reasoning stages:

- query classification
- retrieval strategy
- retrieval sufficiency analysis
- context packaging

Each stage supports `off`, `fallback`, or `always`. Structured outputs are
schema-validated, usage/cost is recorded, and failures can fail open.
Packaging reasoning may prioritize or drop known evidence IDs but cannot write
or rewrite evidence text.

| Profile | Context | Rounds | Entity/event embeddings | Classification | Strategy | Analysis | Packaging |
| --- | ---: | ---: | --- | --- | --- | --- | --- |
| `fast` | 2,500 | 1 | off | off | off | off | off |
| `balanced` | 4,000 | 2 | on | fallback | off | fallback | off |
| `quality` | 6,000 | 2 | on | fallback | fallback | always | fallback |
| `max_quality` | 10,000 | 3 | on | always | always | always | always |

These policies only call reasoning when a `ReasoningBackend` is attached. The
LongMemEval adapter used for the v2 full run did not attach one. Thus the latest
run is a deterministic v2 plus embedding run, not a live-LLM reasoning run.

### ActiveGraph integration and durability

`GraphMemoryRepository` can materialize claims, entities, events, states,
preferences, quantities, temporal references, proofs, and measured retrieval
stages as ActiveGraph objects with stable logical keys. SQLite and graph-backed
embedding stores are available.

The LongMemEval harness currently uses a different persistence path:

- `query_records.jsonl` is appended and fsynced per completed query.
- `run_events.jsonl` records run/query lifecycle events.
- `run_state.json` and `manifest.partial.json` provide resume pointers.
- `retrieval_records.jsonl` stores exact reader context and retrieval metadata.
- SQLite stores preserve both general and fielded compiled embeddings.

The benchmark does not currently materialize each run into an ActiveGraph event
store. Reconstructing an in-memory `MemoryIndex` directly from graph objects is
also still future work.

## Implementation Commit Sequence

The following `activegraph-memory` commits capture the progression. All were
pushed to `main` before the latest full run.

| Commit | Change |
| --- | --- |
| `bcb1322` | Added compiled memory retrieval runtime |
| `55da1f7` | Made out-of-range temporal references safe |
| `1631082` | Added graph query reducers |
| `4bf6f8e` | Hardened graph query reducers |
| `5090d73` | Expanded compiler, reducers, confidence, and retrieval coverage |
| `406d508` | Hardened temporal and preference retrieval |
| `492cfbd` | Hardened compiled answer packet precision |
| `192fb34` | Built typed profile-driven v2 runtime and benchmarking API |
| `4010cc3` | Batched persistent embedding writes |
| `0bfe62b` | Hardened role, temporal, and aggregate retrieval |
| `71e6683` | Added source-grounded proof retrieval |
| `5984919` | Clarified the structural proof reader contract after the v2 run |

Commit `5984919` is the current package pause point. The latest full run
predates that correction and its retrieval artifact records this exact package
identity:

```text
version:      0.2.0
commit:       71e66835d9a9d359688718e185010b98484cf7a6
content hash: sha256:22de1f4c34195880c2c8b308b2d568741e91b045dee961c5829693e66e300467
dirty:        false
```

The harness has since been hardened to carry this identity into final and
partial manifests and to reject a package identity change during resume.

## Experiment Sequence

Smoke runs were used as build gates, not as unbiased score estimates. Their
variance was high and several 0.96 smoke results were followed by materially
lower full scores.

### Smoke runs

| Run label | Overall | Task average | Abstention |
| --- | ---: | ---: | ---: |
| `agmem-pack-smoke` | 0.76 | 0.8187 | 1.0000 |
| `agmem-fullarch-smoke` | 0.88 | 0.9277 | 1.0000 |
| `agmem-fullarch2-smoke` | 0.94 | 0.9634 | 1.0000 |
| `agmem-graphquery-smoke` | 0.82 | 0.8883 | 1.0000 |
| `agmem-graphquery10k-smoke` | 0.86 | 0.9059 | 1.0000 |
| `agmem-graphqueryfix10k-smoke` | 0.96 | 0.9762 | 1.0000 |
| `agmem-adaptivepacket-smoke` | 0.92 | 0.9505 | 1.0000 |
| `agmem-adaptivepacket2-smoke` | 0.94 | 0.9634 | 1.0000 |
| `agmem-adaptivepacket3-smoke` | 0.94 | 0.9634 | 1.0000 |
| `agmem-adaptivepacket4-smoke` | 0.96 | 0.9762 | 1.0000 |
| `agmem-stategraph-smoke` | 0.94 | 0.9634 | 1.0000 |
| `agmem-hardened-smoke` | 0.92 | 0.9078 | 1.0000 |
| `agmem-hardened-10k-smoke` | 0.90 | 0.8959 | 1.0000 |
| `agmem-hardened-10k-smoke2` | 0.96 | 0.9753 | 1.0000 |
| `agmem-hardened-gated-10k-smoke` | 0.96 | 0.9753 | 1.0000 |
| `agmem-v2-maxquality-smoke` | 0.94 | 0.9515 | 1.0000 |
| `agmem-v2-hardened-smoke` | 0.82 | 0.7973 | 1.0000 |
| `agmem-v2-proof-smoke` | 0.96 | 0.9753 | 1.0000 |

### Full 500 runs

| Run label | Overall | Task average | Abstention |
| --- | ---: | ---: | ---: |
| `agmem-fullarch-full` | 0.878 | 0.8943 | 0.9667 |
| `agmem-graphqueryfix10k-full` | 0.810 | 0.8284 | 0.9667 |
| `agmem-adaptivepacket4-full` | 0.852 | 0.8621 | 0.9667 |
| `agmem-stategraph-full` | 0.860 | 0.8720 | 0.9333 |
| `agmem-hardened-10k-full` | 0.854 | 0.8650 | 0.9333 |
| `agmem-hardened-gated-10k-full` | 0.850 | 0.8487 | 0.9333 |
| `agmem-v2-proof-full` | 0.824 | 0.8515 | 0.9333 |

The sequence shows why smoke alone is not sufficient. Architecture quality and
small-set score did not move monotonically together, and the frozen smoke set
did not represent the full density of multi-session and temporal cases.

## Latest Full Run

Run directory:

```text
runs/agmem-v2-proof-full-20260710T015130Z__activegraph-memory-pack__s__full
```

### QA score

| Question type | Correct | Wrong | N | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| single-session-user | 66 | 4 | 70 | 0.9429 |
| single-session-preference | 25 | 5 | 30 | 0.8333 |
| single-session-assistant | 53 | 3 | 56 | 0.9464 |
| multi-session | 93 | 40 | 133 | 0.6992 |
| temporal-reasoning | 105 | 28 | 133 | 0.7895 |
| knowledge-update | 70 | 8 | 78 | 0.8974 |
| **Overall** | **412** | **88** | **500** | **0.8240** |

Relative to the historical `0.878` result, v2 lost 27 correct answers:

| Question type | Historical correct | v2 correct | Delta |
| --- | ---: | ---: | ---: |
| single-session-user | 67 | 66 | -1 |
| single-session-preference | 25 | 25 | 0 |
| single-session-assistant | 55 | 53 | -2 |
| multi-session | 105 | 93 | -12 |
| temporal-reasoning | 112 | 105 | -7 |
| knowledge-update | 75 | 70 | -5 |

### Retrieval and reader diagnostic

Answer-in-context (AIC) excludes the 30 abstention questions and compares
selected source IDs with the benchmark's gold answer locations.

| System | Exact turn hit | Session hit | Exact-turn misses | Reader failed with exact evidence | Correct / 470 |
| --- | ---: | ---: | ---: | ---: | ---: |
| deterministic embedding | 0.8617 | 0.9489 | 65 | 37 | 396 |
| semantic hybrid | 0.9000 | 0.9766 | 47 | 37 | 410 |
| typed v2 proof | 0.9043 | 0.9787 | 45 | 63 | 384 |

`Reader failed with exact evidence` means the answer was judged wrong while at
least one gold source turn was selected. Exact-turn misses include some answers
the reader still got right through redundant or inferential evidence, so the
columns are diagnostics rather than a partition of all errors.

The per-type AIC result sharpens the diagnosis:

| Type | Exact turn hit | Reader failed with evidence |
| --- | ---: | ---: |
| single-session-user | 0.9844 | 4 |
| single-session-preference | 0.5333 | 2 |
| single-session-assistant | 0.9107 | 0 |
| multi-session | 0.8926 | 28 |
| temporal-reasoning | 0.9291 | 21 |
| knowledge-update | 0.9583 | 8 |

Multi-session and temporal retrieval hit rates were high, but the final answer
still failed often. Preference retrieval had a different problem: 96.67%
session hit but only 53.33% exact-turn hit, showing that the right session often
entered context without the best preference turn.

### Performance and cost telemetry

| Metric | Value |
| --- | ---: |
| Wall clock | 11,426.516 s |
| Mean end-to-end query time | 20.604 s |
| Mean retrieval latency | 14,720.42 ms |
| Retrieval p95 latency | 23,660.83 ms |
| Mean context tokens | 9,878.50 |
| Mean prompt tokens | 11,321.03 |
| Mean completion tokens | 104.82 |
| Contexts at budget/truncated | 487 / 500 |
| Structurally proof-complete | 283 / 500 (0.566) |
| Estimated retrieval embedding cost | $1.45081284 |

The reported retrieval cost uses the configured embedding price and does not
represent total reader plus judge spend. Reasoning cost is zero because no
reasoning backend was attached.

The shared embedding cache recorded 1,577 persistent hits and 1,134 misses in
the final process. The fielded ActiveGraph Memory vector store reported 7,958
hits, 2,893 misses, and 2,893 writes in the final manifest.

## Why The Newer Architecture Scored Lower

### 1. Structural proof was presented as semantic verification

The executor's proof status indicates that required fields such as provenance,
candidate bounds, deduplication, source coverage, or operand dates were present.
It does not establish that entity/category matching, canonicalization, temporal
interpretation, or arithmetic selected the intended facts.

The reader contract nevertheless said to use a `Verified candidate` unless its
sources contradicted it. A plausible but wrong candidate could therefore
override raw evidence. This is the clearest general explanation for improved
source recall paired with 26 additional reader failures with exact evidence.

The code now uses `Proof-complete candidate` and tells readers to check every
candidate. This is a general epistemic correction, not a benchmark-specific
rule.

### 2. More context did not guarantee cleaner evidence

The historical run used a 2,500-token budget and the v2 run used 10,000. V2
still filled the budget on 487 questions. Larger context improved coverage, but
also increased duplicate, adjacent, and semantically related distractors. The
reader had to reconcile raw turns, claim headers, and a compiled packet.

The next packaging design should optimize marginal evidence value, source
diversity, and contradiction visibility rather than treating token budget as a
target to fill.

### 3. Complete-set operators remain sensitive to compilation errors

Counts, sums, temporal windows, and negative existence need a well-defined
universe of candidate facts. A structurally bounded scan can still be wrong if:

- an event was not extracted;
- two mentions were merged incorrectly or failed to merge;
- a broad category included an adjacent but irrelevant item;
- a role or modality was misclassified;
- event time was confused with observation time;
- a quantity was attached to the wrong event;
- a source turn existed but was not selected into the final packet.

Proof completion must eventually be calibrated per operator against actual
correctness, with separate confidence for extraction, entity resolution,
coverage, and execution.

### 4. `max_quality` did not exercise its reasoning stages

The profile names configure permission and policy; they do not instantiate an
LLM backend. The benchmark adapter passed an embedding provider but no reasoner.
Classification, strategy, sufficiency analysis, and packaging therefore used
their deterministic paths. Live reasoning may help with ambiguous queries and
evidence sufficiency, but it still needs ablation because it can add latency,
cost, and distractors.

### 5. Preference selection needs exact evidence, not just the right session

The preference AIC gap shows that session-level retrieval can look healthy while
the actual preference expression is absent. Preference memories need explicit
scope, positive/negative polarity, constraints, and links from recommendations
to the profile evidence that justifies them.

## Benchmark-Safe General Improvements

The following changes are useful for agent memory generally and do not depend
on LongMemEval labels at runtime:

1. Calibrate each typed operator separately. Track structural proof completion,
   semantic correctness, and abstention quality as different variables.
2. Make proof packets evidence-first. Present candidate answers as hypotheses
   with cited rows, competing candidates, and known missing coverage.
3. Use confidence to trigger targeted queries. Low entity match, missing
   operands, weak category bounds, or incomplete source coverage should create
   a new retrieval round with an explicit purpose.
4. Make packaging query-aware. Allocate evidence across operands, time periods,
   source roles, and sessions; penalize duplicate semantic content.
5. Preserve a compact fact ledger. Facts, quantities, dates, categories,
   supersession, and canonical event IDs should be queryable independently of
   prose rendering.
6. Separate extraction confidence, source authority, belief confidence,
   retrieval confidence, and execution confidence.
7. Add contradiction and alternate-hypothesis visibility before asking a reader
   to synthesize an answer.
8. Measure live reasoning profiles on the same fixed corpus with stage-level
   latency, token usage, cost, and quality deltas.

## Implemented But Not Yet Fully Benchmarked

- `fast`, `balanced`, `quality`, and `max_quality` profile switches
- per-stage `off` / `fallback` / `always` reasoning policies
- ActiveGraph LLM reasoning backend integration
- schema-validated reasoned query variants
- reasoned sufficiency-triggered retrieval expansion
- evidence-ID-only reasoned packaging
- graph-backed embedding persistence
- graph materialization of compiled memory, proofs, and telemetry
- package-level cold/warm profile benchmark API
- post-run proof-label correction

The committed offline package benchmark uses deterministic hash embeddings and
no live reasoner. It validates control-plane overhead and telemetry plumbing,
not real provider latency or LongMemEval quality differences between profiles.

## Remaining Product Gaps

- provider-neutral typed extraction contract
- extraction coverage calibration against raw source scans
- incremental compilation when new source events arrive
- direct accepted-`memory_gateway` item integration
- typed subject/property identity for supersession
- explicit contradiction objects and resolution policy
- source authority, extraction confidence, and belief confidence separation
- bitemporal valid-time and transaction-time queries
- bounded negative-existence certificates
- relational multi-hop joins
- unit and currency conversion through an explicit provider
- query-aware identity for multiple lists in one source
- reconstruction of `MemoryIndex` from materialized graph objects
- concurrent retrieval attempt identity
- deletion and retention semantics for stored vectors
- live profile speed/cost/quality benchmark table

## Artifact Retention

The repository commits lightweight authoritative evidence for this pause point:

- `scores.json` and `manifest.json` for each attempted ActiveGraph Memory run
- latest full-run hypotheses and upstream judge labels
- latest full-run `query_records.jsonl`
- latest full-run `aic_results.json`

The exact latest `retrieval_records.jsonl` is approximately 38 MB and the local
set of generated ActiveGraph Memory run directories is approximately 84 MB.
Those raw traces remain local to avoid permanently tripling repository size.
Their schema and exact run directory are documented above. Future publication
can move them to a release asset or object store with checksums.

## Reproduction

Offline package and harness gates:

```bash
cd ../activegraph-memory
python3.11 -m pytest -q

cd ../activegraph-longmemeval
make tests PY=python3.11
```

Smoke using the v2 configuration:

```bash
python3.11 -m activegraph_lme.cli run \
  --system activegraph-memory-pack \
  --dataset s \
  --config config/run.activegraph-memory-v2.yaml \
  --run-id agmem-v2-proof-contract-smoke-YYYYMMDDTHHMMSSZ \
  --smoke
```

Full run after a successful smoke:

```bash
python3.11 -m activegraph_lme.cli run \
  --system activegraph-memory-pack \
  --dataset s \
  --config config/run.activegraph-memory-v2.yaml \
  --run-id agmem-v2-proof-contract-full-YYYYMMDDTHHMMSSZ \
  --resume \
  --require-authoritative-tokens
```

Evaluate with the frozen judge:

```bash
python3.11 -m activegraph_lme.cli eval --run-dir runs/<run-directory>
```

Then derive AIC directly from retrieval artifacts:

```bash
python3.11 scripts/answer_in_context.py runs/<run-directory>
```

## Pause-Point Decision

The v2 architecture should be retained. The score regression does not justify
returning to generic top-k retrieval because v2 preserves better structure,
provenance, durability, observability, and raw answer-source recall. It does
show that typed computation and proof labels must be calibrated before readers
are encouraged to trust them.

The next benchmark should be a smoke run of the corrected proof contract. If it
does not regress, the next full run should compare against both `0.878` QA and
the v2 AIC split. Live reasoning/profile cost ablations should follow as
separate cells rather than being mixed into that correction run.
