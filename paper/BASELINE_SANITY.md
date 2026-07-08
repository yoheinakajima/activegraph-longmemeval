# Baseline sanity checks

A short, paper-grade gut-check list. Read this **before** ActiveGraph
enters the matrix. If any of the qualitative orderings below are
violated on `make baselines-smoke`, something is silently broken in the
harness — fix it before trusting later numbers.

## Expected qualitative ordering

1. **`full-context-oracle` ≥ every other system on accuracy.**
   Oracle feeds only the labeled-evidence sessions, so it is the
   functional upper bound under our reader. If any retrieval system
   *beats* oracle on overall accuracy, either (a) the reader is leaking
   information across calls, (b) the oracle isn't actually receiving the
   evidence sessions, or (c) the judge is being asked to score an
   abstention prompt incorrectly. Investigate before publishing.

2. **`full-context-oracle` ≥ `full-context-s` on accuracy.**
   Same model, same prompt: stuffing 100+ irrelevant sessions can only
   hurt or tie. A reversed ordering means full-context-s is losing
   evidence sessions to truncation (check `n_truncated` in the manifest
   and the `mean_context_tokens`) or oracle is malformed.

3. **`full-context-s` ≥ `rag-*` on most question types, with notable
   exceptions for `temporal-reasoning` and `multi-session`** where
   irrelevant sessions in the stuffed context act as distractors and
   retrieval-based methods can win. Expect RAG to outperform stuffing
   here but trail on `single-session-*` types.

4. **`rag-dense` ≥ `rag-bm25` on the semantic types**
   (`multi-session`, `temporal-reasoning`, `knowledge-update`).
   BM25 typically holds its ground on `single-session-user` (lexical
   overlap with the user's recent message). If dense loses across the
   board, the embedding pipeline is suspect (wrong model, wrong
   pooling, broken normalization).

5. **Turn-granularity ≥ session-granularity on `single-session-*` types;
   session-granularity ≥ turn-granularity on `multi-session`.**
   This is the granularity sweep upstream LongMemEval highlights. If
   the relationship is flat or inverted everywhere, the granularity
   knob isn't actually wired through.

6. **Abstention accuracy is high for `full-context-oracle` only when
   the prompt actually allows "I don't know".** All other systems can
   only hit abstention if the *retrieved* context omits the (absent)
   answer; their abstention numbers should be comparable to oracle's.
   Wildly low abstention across the board usually means our prompt is
   forcing a guess.

## Mechanical checks (already automated)

- `aglme run` runs a per-system determinism check
  (`retrieve(state, q, d)` called twice; texts must match) on
  the first instance of every run.
- `scripts/property_tests.py` (`make tests`) asserts: oracle excludes
  distractors, BM25 ranks the evidence session first on a planted
  query, the RAG granularity axis produces distinct outputs, and
  full-context-s truncation drops oldest first and sets the
  `truncated` flag.
- The judge wrapper asserts hypothesis ↔ reference `question_id` sets
  match before invoking the upstream eval; a silent ID mismatch can no
  longer corrupt scores.
- Manifest records `context_token_source` ∈ {`tiktoken`,
  `charfallback`}; paper runs fail unless it equals `tiktoken`.

## Numbers we are NOT trying to reproduce

Upstream LongMemEval reports numbers against Llama-3-as-reader (or
similar). We use Claude Sonnet at temperature 0. **Absolute accuracy
levels will differ.** The *relative* ordering above is what we verify.

## ActiveGraph deterministic (Mode A)

Two sub-variants run alongside the four baselines:
`activegraph-det-lexical` and `activegraph-det-embedding`. Both build
the same deterministic Turn graph (no LLM at ingest) and assemble under
the same `activegraph.token_budget` (config default 2500 — mirrors the
turn-level RAG baselines so accuracy comparisons aren't confounded by
context size).

Expectations:

- Both ActiveGraph variants should land *between* the best turn-level
  RAG baseline and `full-context-oracle` at comparable mean context
  tokens. If either *beats* oracle, treat it as a bug — oracle is the
  upper bound under this reader.
- `activegraph-det-embedding` should ≥ `activegraph-det-lexical` on
  semantic question types (`multi-session`, `temporal-reasoning`,
  `knowledge-update`) for the same reason `rag-dense` typically beats
  `rag-bm25` on those types.
- `activegraph-det-lexical` should be competitive on
  `single-session-user` (lexical overlap with the user's message) and
  may equal or exceed the embedding sub-variant there.
- `n_truncated` per the manifest will be 0 for most haystacks at the
  default budget; if it's non-zero on a large fraction of questions
  the budget needs to be raised (and reported alongside the new
  comparison baseline).

## ActiveGraph Memory Pack Adapter

`activegraph-memory-pack` is a Phase 1 integration cell for the external
`activegraph-memory` repository. In v0.1 it should be treated as
instrumentation over the deterministic lexical context, not as an expected
accuracy improvement. Its first sanity check is mechanical:

- `make tests` should report the adapter contract as `OK` when
  `activegraph-memory` is installed or available as a sibling checkout.
- The adapter should emit `retrieval_plan`, `coverage_report`,
  `confidence`, and `gateway_request` metadata while keeping the reader
  context deterministic.
- Accuracy lift should only be interpreted after future pack versions alter
  retrieval/assembly through claims, temporal refs, supersession, or
  evidence bundles.
