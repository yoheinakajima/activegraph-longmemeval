# ActiveGraph Memory v4 Experiment Report

Report date: 2026-07-10

## Experiment Bundle

v4 combines five of the seven next experiments from the v3 report under one
causal hypothesis: memory quality improves when the system measures evidence
completeness across the pipeline and packages required facts by role rather
than adding more weakly ranked context.

Included:

1. query-bounded extraction, compilation, selection, and reader-recovery
   coverage accounting;
2. exact scoped preference retrieval with positive/negative polarity;
3. slot-based packaging for aggregate events, temporal operands, preferences,
   constraints, and raw recovery sources;
4. per-operator confidence floors plus a held-out calibration API;
5. non-benchmark finance, project-state, scheduling, preference, and agent-task
   traces.

Deferred:

- live LLM reasoning-stage ablations, because they change model cost and latency;
- cold-cache profiling, because it measures infrastructure cost rather than
  evidence quality.

The deferred experiments remain useful, but mixing them into this run would
make attribution weaker.

## Full-Run Policy

v4 does not run a full 500 after each subexperiment. The bundle receives:

1. package unit and integration tests;
2. deterministic profile benchmarks on the existing and non-benchmark fixtures;
3. the LongMemEval offline adapter contract;
4. one frozen 50-question smoke;
5. at most one full 500 only if the smoke clears the gate.

The smoke gate is overall accuracy at least 0.94, exact gold-turn recall at
least 0.93 on non-abstention questions, no severe task-type collapse, complete
0.4.0 package identity, and valid retrieval artifacts. This is a regression
gate, not an estimate of full accuracy.

## Package Identity

```text
version:      0.4.0
commit:       126784675aa9ffc6573b166a88e02ddd7ad45275
content hash: sha256:eb11fa0af8ec903ad2706e06b21008184a23f06b9b9a66fb58738b05a4189d0b
```

## Offline Results

- package tests: 115 passed;
- LongMemEval adapter property tests: passed;
- existing five-case scalar fixture: 1.000 quality for all four profiles;
- v4 five-case application fixture: 1.000 scalar quality for all four profiles;
- application fixture deterministic mean latency: 2.273–2.988 ms by profile;
- application fixture mean coverage confidence: 1.000;
- application fixture mean evidence slots: 1.6;
- application fixture raw recovery rate: 0.000 on the complete fixture;
- incomplete-extraction tests confirm raw recovery reaches context without
  certifying an incomplete count.

## LongMemEval Results

### Frozen 50-question smoke

Run: `agmem-v4-smoke-20260710T143235Z__activegraph-memory-pack__s__smoke`

| Metric | v3 smoke | v4 smoke |
| --- | ---: | ---: |
| Overall accuracy | 0.9800 | 0.9800 |
| Task-averaged accuracy | 0.9872 | 0.9872 |
| Abstention accuracy | 1.0000 | 1.0000 |
| Exact gold-turn recall, non-abstention | 0.9565 | 0.9565 |
| Gold-session recall, non-abstention | 0.9783 | 0.9783 |
| Proof-complete rate | 0.6200 | 0.7800 |
| Retrieval-sufficient rate | 0.4200 | 0.6000 |
| Candidate packet rendered | 0.2000 | 0.3800 |
| Mean retrieval latency, warm cache | 2,838.573 ms | 2,366.699 ms |
| Mean context tokens | 8,123.92 | 8,119.92 |

The only v4 miss was a multi-session reader failure with all benchmark-marked
turns in context. The two exact-turn misses were preference questions, but both
were answered correctly from other evidence in the retrieved sessions. No task
type collapsed: multi-session was 0.9231 and every other type was 1.0000.

v4 emitted one or more typed evidence slots on 21 of 50 questions, with 2.9
slots per question overall. Seventeen aggregate or preference queries emitted
the new source-coverage audit. Mean compiled selection coverage was 0.6306 and
mean reader-visible coverage was 0.8587; eight audits used bounded raw-source
recovery, and eight met the complete coverage threshold. Recovery improved the
reader packet but did not certify an incomplete computed answer.

The smoke cleared every predefined gate, so it authorized one full 500 run.
The smoke artifacts were generated before correcting a telemetry-only adapter
label that represented package `0.4.0` as runtime `v0`. The manifest's package
version, commit, and content hash were correct, retrieval behavior was
unchanged, and the adapter contract now maps pre-1 package minor versions to
their public architecture series (`0.4.x` to `v4`) before the full run.

### Full 500

Run: `agmem-v4-full-20260710T144623Z__activegraph-memory-pack__s__full`

| Run | Correct | Overall | Task average | Abstention |
| --- | ---: | ---: | ---: | ---: |
| Historical compiled-memory best | 439 | 0.8780 | 0.8943 | 0.9667 |
| Typed v3 calibrated | 416 | 0.8320 | 0.8508 | 0.9667 |
| Typed v4 evidence quality | 416 | 0.8320 | 0.8525 | 0.9667 |

v4 tied v3 overall. It produced 18 paired gains and 18 paired losses; 398
questions were correct in both runs and 66 were wrong in both. The task
average rose by 0.0017 because the zero-net changes moved between task types.

| Type | N | v3 correct | v4 correct | v4 accuracy | Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| single-session-user | 70 | 67 | 67 | 0.9571 | 0 |
| single-session-preference | 30 | 23 | 23 | 0.7667 | 0 |
| single-session-assistant | 56 | 52 | 53 | 0.9464 | +1 |
| multi-session | 133 | 94 | 97 | 0.7293 | +3 |
| temporal-reasoning | 133 | 106 | 102 | 0.7669 | -4 |
| knowledge-update | 78 | 74 | 74 | 0.9487 | 0 |

### Retrieval And Reader Decomposition

Answer-in-context analysis excludes the 30 abstention questions.

| Metric | v3 | v4 | Delta |
| --- | ---: | ---: | ---: |
| Exact gold-turn recall | 0.8979 | 0.8915 | -0.0064 |
| Gold-session recall | 0.9723 | 0.9766 | +0.0043 |
| Exact-turn misses | 48 | 51 | +3 |
| Session misses | 13 | 11 | -2 |
| Reader failures with every gold turn present | 57 | 55 | -2 |
| Non-abstention correct | 387 | 387 | 0 |

v4 had no question where an exact-turn miss in v3 became an exact-turn hit. It
introduced three exact-turn regressions: one preference, one aggregate sum,
and one temporal order question. It did recover three session-level misses and
introduced one session-level regression. The evidence packaging helped the
reader on some questions, but did not improve raw exact-evidence recall.

Only 268 of 500 rendered contexts were byte-identical between v3 and v4, and
149 final answers were byte-identical. Of the 18 gains, four used identical
contexts; of the 18 losses, five used identical contexts. This confirms that a
small portion of paired movement is reader variance even with a pinned model
and temperature zero. Most movement came from changed compiled packets.

### Coverage And Calibration

| Gate | v3 rate | v4 rate | v3 positive accuracy | v4 positive accuracy |
| --- | ---: | ---: | ---: | ---: |
| Structural proof complete | 0.5660 | 0.6820 | 0.8763 | 0.8651 |
| Retrieval sufficient | 0.4180 | 0.5380 | 0.8852 | 0.8662 |
| Candidate packet rendered | 0.2040 | 0.3220 | 0.8824 | 0.8509 |

v4 made more queries structurally complete and reader-ready, but the expanded
positive groups were less accurate. The gates remain directionally useful:
v4 proof-complete queries scored 0.8651 versus 0.7610 when incomplete, and
sufficient queries scored 0.8662 versus 0.7922 when insufficient. They are not
answer-correctness certificates.

Typed evidence slots were present on 233 questions, averaging 3.738 slots per
question over the full set. The new source-coverage audit ran on 183 aggregate
or preference questions:

| Audit metric | Value |
| --- | ---: |
| Mean extraction ratio | 0.5703 |
| Mean compilation ratio | 0.5834 |
| Mean selected-source ratio | 0.4719 |
| Mean reader-visible ratio | 0.8703 |
| Complete audits | 57 / 183 |
| Audits using raw-source recovery | 114 / 183 |

The audit discriminates real risk. Complete audits had 0.8421 answer accuracy
and 0.9808 exact-turn recall; incomplete audits had 0.7778 accuracy and 0.8067
exact-turn recall. Recovery cases had 0.7719 accuracy and 0.7890 exact-turn
recall. Raw recovery raised reader-visible coverage, but it did not reliably
select the exact missing evidence and correctly did not certify incomplete
computed answers.

### Operator Results

| Operator | N | v3 accuracy | v4 accuracy | Delta | v4 proof | v4 exact-turn recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lookup | 149 | 0.8926 | 0.8859 | -1 answer | 0.9933 | 0.9420 |
| count | 117 | 0.7949 | 0.8120 | +2 | 0.2222 | 0.9352 |
| previous | 48 | 0.8958 | 0.9167 | +1 | 1.0000 | 0.9167 |
| order | 40 | 0.8250 | 0.7250 | -4 | 0.6750 | 0.8421 |
| sum | 40 | 0.8250 | 0.8500 | +1 | 0.5500 | 0.9459 |
| snapshot-count | 38 | 0.8158 | 0.7632 | -2 | 1.0000 | 0.8571 |
| date-delta | 22 | 0.7727 | 0.8182 | +1 | 0.6364 | 1.0000 |
| preference profile | 17 | 0.5882 | 0.6471 | +1 | 0.0000 | 0.2353 |
| current | 16 | 0.9375 | 0.9375 | 0 | 0.9375 | 0.8571 |
| max | 9 | 0.6667 | 0.6667 | 0 | 0.1111 | 0.7778 |
| ordinal | 4 | 0.5000 | 0.7500 | +1 | 0.5000 | 0.5000 |

Count, sum, date-delta, previous-state, preference, and ordinal answers gained
eight correct answers in total. Lookup, snapshot-count, and order lost seven;
one additional max gain and one max loss canceled. The largest regression was
temporal order, despite its proof-complete rate rising from 0.2000 to 0.6750.

Inspection of the order losses found general implementation problems:

1. multi-operand parsing sometimes treated an entire enumerated question as
   one operand instead of splitting each event;
2. slot grounding accepted semantically related turns that did not state the
   requested event;
3. ordering used observation dates when a source expressed an earlier relative
   event time such as "a month ago";
4. same-day events were ordered without reliable turn-level chronology; and
5. a structurally complete but incorrectly grounded candidate biased the
   reader more strongly than the raw source evidence.

These are generic event-time and evidence-grounding defects, not benchmark
exceptions. They explain why more proof packets did not produce higher QA.

Scoped preference retrieval also remains incomplete. Preference-profile
accuracy gained one answer, but exact-turn recall fell from 0.2941 to 0.2353.
All 17 preference audits were incomplete and used recovery. The polarity and
scope representation is useful, but candidate discovery still does not reach
the exact positive or negative preference statement often enough.

### Performance And Cost

| Metric | v3 | v4 |
| --- | ---: | ---: |
| Reader-pass active wall clock | 6,144.388 s | 6,873.091 s |
| Mean retrieval latency | 2,854.020 ms | 2,702.699 ms |
| Retrieval p95 latency | 4,027.306 ms | 3,861.507 ms |
| Mean context tokens | 8,117.43 | 8,123.24 |
| Mean prompt tokens | 9,318.64 | 9,326.04 |
| Mean completion tokens | 114.07 | 117.47 |
| Truncated contexts | 488 / 500 | 488 / 500 |
| Mean retrieval rounds | 2.010 | 1.792 |
| Retrieval-round distribution | 202 / 91 / 207 | 264 / 76 / 160 |
| Estimated retrieval embedding cost | $0.00036992 | $0.00097304 |

The v4 retrieval path was 5.3% faster on mean and 4.1% faster at p95 while
using fewer adaptive rounds. Context size was effectively unchanged. The
retrieval-cost difference is sub-cent and reflects cache misses and query
embedding work, not optional reasoning calls; live reasoning remained off.

The full run also exercised recovery under real failure. A provider request
stalled after question 236, the process was stopped, and `--resume` continued
at 237 without duplicating completed records. A later process interruption at
490 resumed only the final ten. The sealed run contains exactly 500 unique
hypotheses, query records, retrieval records, and manifest query entries. The
Anthropic client currently relies on the SDK's long default retry window;
explicit bounded request timeouts are a harness reliability improvement for
the next run.

## Conclusions

1. v4 delivered the intended product capabilities: query-bounded coverage
   accounting, typed evidence slots, scoped preference polarity, per-operator
   thresholds, held-out calibration APIs, raw recovery, and non-benchmark
   application traces all exist with deterministic tests.
2. It did not improve LongMemEval overall accuracy. The full result tied v3 at
   0.832 and remains 23 answers below the historical 0.878 run.
3. Coverage auditing is valuable even without a score gain. It identifies
   incomplete evidence sets and prevents recovery text from falsely certifying
   aggregate answers.
4. Packaging more structured evidence is harmful when slot grounding or event
   time is wrong. Proof must require exact predicate grounding and resolved
   event time, not merely a source, entity similarity, and observation date.
5. Retrieval remains responsible for 28 wrong non-abstention answers without
   every exact turn; the reader remains responsible for 55 wrong answers with
   every exact turn present. Both layers still matter.
6. Smoke remained a regression gate, not a score estimator: its 0.98 result
   overstated full accuracy by 14.8 points again.

## Next Work

Do not run another full 500 until these changes pass deterministic application
traces and the same frozen smoke:

1. parse comparison and ordering questions into explicit operands using a
   structured query representation that handles lists, conjunctions, quoted
   events, and repeated entity names;
2. compile resolved event time separately from observation time, propagate
   relative-date uncertainty, and refuse an order candidate when event time is
   unresolved;
3. require each slot source to match the requested entity, predicate, role,
   and polarity, with embedding similarity used for candidate generation but
   never as proof of slot satisfaction;
4. connect recovery sources to selected evidence IDs and rerun compilation so
   recovered exact evidence can improve the computation, while preserving the
   rule that reader-only recovery cannot certify it;
5. package temporal operands as a compact table of requested operand, resolved
   event date, minimal supporting excerpt, and source ID; suppress candidate
   ordering when any row lacks grounded time;
6. calibrate thresholds per operator on held-out finance, project, schedule,
   preference, and agent traces, with special attention to order and recommend;
7. add bounded reader request timeouts and deterministic resume tests to the
   benchmark harness; and
8. keep live reasoning-stage and cold-cache experiments separate so their
   quality, latency, and cost effects remain attributable.

## Reproduction

```bash
python3.11 -m activegraph_lme.cli run \
  --system activegraph-memory-pack \
  --dataset s \
  --config config/run.activegraph-memory-v4.yaml \
  --run-id agmem-v4-full-YYYYMMDDTHHMMSSZ \
  --resume \
  --require-authoritative-tokens

python3.11 -m activegraph_lme.cli eval \
  --run-dir runs/<v4-full-run> \
  --config config/run.activegraph-memory-v4.yaml

python3.11 scripts/answer_in_context.py runs/<v4-full-run>
```

## Retained Artifacts

The repository retains the smoke manifest and scores plus the full run's
manifest, scores, hypotheses, judge labels, query telemetry, and answer-in-
context result. The 36 MB raw retrieval contexts, event stream, partial
manifest, run state, and logs remain local operational artifacts; the report's
retrieval statistics are derived from them, while the concise committed files
preserve the authoritative score and provenance.
