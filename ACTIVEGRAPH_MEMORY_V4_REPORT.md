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

Pending.
