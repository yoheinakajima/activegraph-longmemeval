# ActiveGraph Memory v3 LongMemEval Report

Report date: 2026-07-10

Benchmark repository: `yoheinakajima/activegraph-longmemeval`

Memory package repository: `yoheinakajima/activegraph-memory`

## Scope

This report records the first smoke and full LongMemEval runs of
`activegraph-memory` 0.3.0. It follows
`ACTIVEGRAPH_MEMORY_V2_REPORT.md`, which documents the implementation history
and the 0.824 typed-v2 result, and `ACTIVEGRAPH_MEMORY_FULL_RUN_REPORT.md`, which
documents the historical 0.878 best.

The v3 run tests the completed deterministic retrieval path and corrected
reader contract. It does not test live LLM extraction, optional retrieval
reasoning, or graph-store restart inside the benchmark adapter. Those package
features are covered by offline tests and package benchmarks, but require
separate LongMemEval cells to attribute quality, latency, and cost.

## Frozen Evaluation Boundary

| Component | Value |
| --- | --- |
| Dataset | `data/longmemeval_s_cleaned.json` |
| Questions | 500 full; frozen stratified 50 smoke |
| Reader requested | `claude-sonnet-4-5` |
| Reader resolved | `claude-sonnet-4-5-20250929` |
| Reader temperature | 0.0 |
| Reader max output | 1,024 tokens |
| Reader tools/web | disabled |
| Judge | `gpt-4o-2024-08-06` through upstream `gpt-4o` |
| Judge temperature | 0.0 |
| Embeddings | `text-embedding-3-small` |
| Extraction input | role-aware frozen `seed-A-v2` cache |
| Runtime profile | `max_quality` |
| Context budget | 10,000 tokens |
| Optional reasoner | none |
| Token accounting | API usage for reader; tiktoken for context |

The package identity was locked in both run manifests:

```text
version:      0.3.0
commit:       58f235b9123cca8ce282a3c6c82f2a0f9aa8df2e
content hash: sha256:05c541c63f07a17fc70e5e224af5ce2c78a44c7bd70f50ff35fff2a91fc5649e
dirty:        false
```

## What Changed Since The v2 Full Run

The 0.3 package adds or completes:

- provider-neutral typed fact extraction and a deterministic raw-turn fallback;
- bounded model-extraction batches with per-batch and aggregate usage;
- direct typed compiler inputs for entities, aliases, predicates, categories,
  modality, polarity, event time, quantities, state identity, preference scope,
  validity, and confidence;
- separate extraction and belief confidence;
- unresolved conflict objects and conflict-aware candidate suppression;
- deterministic retrieval sufficiency assessment across recall, entity
  resolution, source coverage, temporal resolution, consistency, and execution;
- confidence-driven targeted query rounds with explicit stop reasons;
- calibrated candidate rendering instead of presenting every structurally
  complete computation as verified;
- bounded negative-existence certificates that do not claim world-level absence;
- cumulative reasoning stop thresholds and independently configurable
  classification, strategy, analysis, and packaging reasoning;
- replayable source turns, ingestion runs, claims, compiled projections,
  retrieval assessments, proofs, and stage telemetry in ActiveGraph;
- graph-to-index load after restart and idempotent append/recompile;
- persistent SQLite or graph-backed corpus embeddings;
- deterministic profile, option-matrix, reasoning-ablation, and ingestion
  benchmarks with latency, tokens, cost, proof, sufficiency, and quality fields.

The LongMemEval adapter uses the 0.3 query/retrieval/execution/packaging runtime,
fielded embeddings, and persistent vector cache. It still compiles the frozen
extraction cache directly into an in-process index. Therefore this run does not
measure the new LLM extractor, graph materialization, or restart loader.

## Smoke Gate

Run directory:

```text
runs/agmem-v3-smoke-20260710T070604Z__activegraph-memory-pack__s__smoke
```

| Metric | Result |
| --- | ---: |
| Overall accuracy | 0.9800 (49/50) |
| Task-averaged accuracy | 0.9872 |
| Abstention accuracy | 1.0000 (4/4) |
| Exact gold-turn recall, non-abstention | 0.9565 (44/46) |
| All gold sessions present | 0.9783 (45/46) |
| Structurally proof-complete | 0.6200 |
| Deterministically sufficient | 0.4200 |
| Candidate rendered | 0.2000 |
| Mean retrieval rounds | 1.96 |

Per-type smoke accuracy:

| Type | Accuracy | N |
| --- | ---: | ---: |
| single-session-user | 1.0000 | 7 |
| single-session-preference | 1.0000 | 3 |
| single-session-assistant | 1.0000 | 5 |
| multi-session | 0.9231 | 13 |
| temporal-reasoning | 1.0000 | 14 |
| knowledge-update | 1.0000 | 8 |

The only miss was `gpt4_2f8be40d`, a count of weddings. Retrieval selected all
three exact gold turns and the compiled executor correctly returned `3`. The
reader named the wedding relationships and venues but omitted the requested
couple names. The judge rejected the answer against the gold response naming
Rachel and Mike, Emily and Sarah, and Jen and Tom. This was not a retrieval or
arithmetic failure; it was evidence-detail loss during answer synthesis.

The 0.98 smoke cleared the full-run gate, but the historical sequence already
showed that the frozen smoke overestimates full multi-session density. It was a
regression check, not an estimate of full accuracy.

## Full 500 Result

Run directory:

```text
runs/agmem-v3-full-20260710T071914Z__activegraph-memory-pack__s__full
```

| Metric | Result |
| --- | ---: |
| Correct | 416 |
| Wrong | 84 |
| Overall accuracy | 0.8320 |
| Task-averaged accuracy | 0.8508 |
| Abstention accuracy | 0.9667 (29/30) |

Per-type results:

| Type | Correct | Wrong | N | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| single-session-user | 67 | 3 | 70 | 0.9571 |
| single-session-preference | 23 | 7 | 30 | 0.7667 |
| single-session-assistant | 52 | 4 | 56 | 0.9286 |
| multi-session | 94 | 39 | 133 | 0.7068 |
| temporal-reasoning | 106 | 27 | 133 | 0.7970 |
| knowledge-update | 74 | 4 | 78 | 0.9487 |

### Comparison

| Run | Correct | Overall | Task average | Abstention |
| --- | ---: | ---: | ---: | ---: |
| Historical compiled-memory best | 439 | 0.878 | 0.8943 | 0.9667 |
| Typed v2 proof | 412 | 0.824 | 0.8515 | 0.9333 |
| Typed v3 calibrated | 416 | 0.832 | 0.8508 | 0.9667 |

Relative to v2, v3 gained four correct answers overall. It gained one
single-session-user, one multi-session, one temporal, and four knowledge-update
answers, while losing two preference and one assistant answer. One additional
abstention was correct. The task-average is effectively flat because the gains
and losses are distributed unevenly across task types.

Relative to the historical 0.878 run, v3 remains 23 answers behind:

| Type | Historical correct | v3 correct | Delta |
| --- | ---: | ---: | ---: |
| single-session-user | 67 | 67 | 0 |
| single-session-preference | 25 | 23 | -2 |
| single-session-assistant | 55 | 52 | -3 |
| multi-session | 105 | 94 | -11 |
| temporal-reasoning | 112 | 106 | -6 |
| knowledge-update | 75 | 74 | -1 |

## Retrieval And Reader Decomposition

Answer-in-context analysis excludes the 30 abstention questions. Exact-turn
hit requires all benchmark-marked answer turns to be selected.

| Type | N | Exact-turn hit | Session hit | Turn misses | Reader failures with exact evidence | Non-abstention accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| knowledge-update | 72 | 0.9583 | 1.0000 | 3 | 4 | 0.9444 |
| multi-session | 121 | 0.8843 | 0.9752 | 14 | 26 | 0.6860 |
| single-session-assistant | 56 | 0.9107 | 1.0000 | 5 | 1 | 0.9286 |
| single-session-preference | 30 | 0.5333 | 0.9333 | 14 | 3 | 0.7667 |
| single-session-user | 64 | 0.9844 | 1.0000 | 1 | 3 | 0.9531 |
| temporal-reasoning | 127 | 0.9134 | 0.9370 | 11 | 20 | 0.7874 |
| **Overall** | **470** | **0.8979** | **0.9723** | **48** | **57** | **0.8234** |

The contingency table is:

| Exact gold turns selected | Correct | Wrong | N | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| yes | 365 | 57 | 422 | 0.8649 |
| no | 22 | 26 | 48 | 0.4583 |

Compared with v2:

- exact-turn recall decreased from 0.9043 to 0.8979;
- exact-turn misses increased from 45 to 48;
- reader failures despite exact evidence decreased from 63 to 57;
- non-abstention correct answers increased from 384 to 387.

The corrected proof language and calibrated candidate gate improved downstream
use of retrieved evidence, but did not improve raw exact-turn recall. The net
gain is small because six fewer reader-with-evidence failures were partly offset
by three more retrieval misses and task-specific regressions.

Preference remains structurally different from the other failures. Session
recall is 93.33%, but exact preference-turn recall is only 53.33%. The right
conversation is often present without the precise positive/negative preference
or constraint statement needed for the answer.

## Calibration Results

| Gate | State | N | Correct | Accuracy |
| --- | --- | ---: | ---: | ---: |
| Structural proof | complete | 283 | 248 | 0.8763 |
| Structural proof | incomplete | 217 | 168 | 0.7742 |
| Sufficiency assessment | sufficient | 209 | 185 | 0.8852 |
| Sufficiency assessment | insufficient | 291 | 231 | 0.7938 |
| Candidate packet | rendered | 102 | 90 | 0.8824 |
| Candidate packet | suppressed | 398 | 326 | 0.8191 |

These gates are directionally calibrated: each positive group is more accurate
than its negative group. They are not correctness certificates. Twelve of 102
rendered candidates still led to wrong final answers, and 24 of 209
deterministically sufficient packets were wrong.

The candidate-render rate is 20.4%, well below the 56.6% structural proof rate.
This is intentional. A candidate now also requires the sufficiency threshold
and no unresolved selected conflict; structural proof alone is not enough.

## Operator Diagnostics

| Primary operator | N | Accuracy | Proof complete | Sufficient | Exact-turn hit |
| --- | ---: | ---: | ---: | ---: | ---: |
| count | 155 | 0.8000 | 0.2645 | 0.1935 | 0.9161 |
| lookup | 149 | 0.8926 | 0.9933 | 0.7047 | 0.9420 |
| previous | 48 | 0.8958 | 1.0000 | 0.8333 | 0.9167 |
| order | 40 | 0.8250 | 0.2000 | 0.2000 | 0.8684 |
| sum | 40 | 0.8250 | 0.1250 | 0.1250 | 0.9730 |
| date_delta | 22 | 0.7727 | 0.0909 | 0.0909 | 1.0000 |
| recommend | 17 | 0.5882 | 0.8235 | 0.1176 | 0.2941 |
| current | 16 | 0.9375 | 0.9375 | 0.9375 | 0.8571 |
| max | 9 | 0.6667 | 0.0000 | 0.0000 | 0.7778 |
| ordinal | 4 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |

The table should not be interpreted as benchmark-specific operator accuracy:
question classes and operator assignments overlap imperfectly, and some groups
are small. It does expose general engineering priorities:

1. Complete-set operators often lack source-coverage proof even when the answer
   is correct. Count, sum, max, and order need stronger coverage retrieval and
   extraction-recall accounting.
2. Date-delta retrieval is excellent, but execution/synthesis remains fallible;
   all exact turns were selected while accuracy was 77.27%.
3. Recommendation proof fields do not imply exact profile evidence. The low
   exact-turn hit and low sufficiency rate show that scope and constraint
   retrieval need improvement.
4. Lookup, previous-state, and current-state paths are the strongest and most
   consistently calibrated.

## Performance And Cost

| Metric | Value |
| --- | ---: |
| Reader-pass wall clock | 6,144.388 s |
| Mean retrieval latency | 2,854.020 ms |
| Retrieval p95 latency | 4,027.306 ms |
| Mean context tokens | 8,117.43 |
| Mean prompt tokens | 9,318.64 |
| Mean completion tokens | 114.07 |
| Contexts marked truncated | 488 / 500 |
| Mean retrieval rounds | 2.01 |
| Round distribution | 202 one; 91 two; 207 three |
| Structural proof-complete rate | 0.5660 |
| Sufficiency rate | 0.4180 |
| Candidate-render rate | 0.2040 |
| Estimated retrieval embedding cost | $0.00036992 |
| Optional reasoning calls | 0 |

The retrieval-cost estimate covers embeddings at the configured price. It does
not include reader or judge spend. The large reduction from v2's reported
$1.45081284 retrieval cost and 14.7-second mean retrieval latency is primarily
the effect of persistent warm embedding caches, not a clean cold-start runtime
comparison.

## Conclusions

1. The 0.3 architecture is materially more truthful and operationally complete
   than v2: graph replay, typed ingestion, conflicts, confidence-driven rounds,
   calibrated candidates, restart-safe telemetry, and explicit cost controls
   are real package capabilities with deterministic tests.
2. The corrected proof contract helped but did not recover the historical
   0.878 score. Full accuracy improved from 0.824 to 0.832, while exact-turn
   recall slightly declined.
3. Retrieval remains the dominant opportunity for 26 non-abstention misses,
   while answer synthesis remains the dominant opportunity for 57 misses where
   every exact gold turn was already present.
4. Multi-session and temporal questions account for 66 of 84 total misses.
   This is consistent with set coverage, event identity, temporal semantics,
   and evidence synthesis being the hard general memory problems.
5. Preference retrieval needs exact scoped evidence rather than session-level
   similarity. Generic recommendation proof is not sufficiently selective.
6. Smoke accuracy is not a reliable estimator of full accuracy. The 0.98 smoke
   was useful as a regression gate but overstated the full score by 14.8 points.
7. Live reasoning remains unmeasured. It should be tested as explicit ablations
   with provider usage attached, not enabled wholesale and attributed to the
   architecture.

## Next Experiments

The next work should remain benchmark-independent:

1. Add retrieval coverage accounting that estimates extraction and index
   completeness, not only whether the current candidate scan was bounded.
2. Improve exact preference/constraint retrieval with typed scope, polarity,
   exclusions, and links from recommendation candidates to profile evidence.
3. Package multi-operand and aggregate evidence by required entity/event/time
   slots so the reader cannot omit names or conflate semantically similar rows.
4. Calibrate candidate and sufficiency thresholds per operator on held-out
   application traces as well as LongMemEval.
5. Run reasoning ablations for classification, strategy, analysis, packaging,
   and all stages with identical retrieval budgets and recorded provider cost.
6. Add non-benchmark evaluations for project state, finance, scheduling,
   preferences, and long-running agent task histories.
7. Run a cold-cache profile benchmark separately from the warm-cache benchmark;
   do not compare the current warm v3 latency/cost directly with cold v2.

## Reproduction

```bash
cd ../activegraph-memory
python3.11 -m pip install -e '.[dev]'
python3.11 -m pytest -q

cd ../activegraph-longmemeval
make tests PY=python3.11

python3.11 -m activegraph_lme.cli run \
  --system activegraph-memory-pack \
  --dataset s \
  --config config/run.activegraph-memory-v2.yaml \
  --run-id agmem-v3-smoke-YYYYMMDDTHHMMSSZ \
  --smoke

python3.11 -m activegraph_lme.cli run \
  --system activegraph-memory-pack \
  --dataset s \
  --config config/run.activegraph-memory-v2.yaml \
  --run-id agmem-v3-full-YYYYMMDDTHHMMSSZ \
  --resume \
  --require-authoritative-tokens

python3.11 -m activegraph_lme.cli eval --run-dir runs/<run-directory>
python3.11 scripts/answer_in_context.py runs/<run-directory>
```

## Retained Artifacts

The repository retains the lightweight authoritative artifacts for both v3
runs: manifests, scores, full hypotheses and judge labels, full query records,
and the full AIC result. Raw retrieval contexts, logs, partial manifests, event
streams, and run-state files remain local because they are large operational
artifacts rather than concise publication evidence.
