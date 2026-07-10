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
commit:       1267846176a24e4c11de375015ad4db881c7f47b
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

Pending the frozen smoke gate.
