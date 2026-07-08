# activegraph-longmemeval

Reproducible benchmark harness for evaluating **ActiveGraph** against the
**LongMemEval** benchmark (ICLR 2025), with a frozen evaluation boundary.

> **Top priority:** every number in the paper must be regenerable from a clean
> clone with one command. We do not reimplement scoring — the official
> [LongMemEval](https://github.com/xiaowu0162/longmemeval) repository is
> vendored as a submodule pinned to a specific commit SHA and called as-is.

## Pinned identities

| | Pin |
|---|---|
| Upstream LongMemEval submodule | `xiaowu0162/longmemeval @ 9e0b455f4ef0e2ab8f2e582289761153549043fc` |
| Python | `3.11` |
| Reader (alias requested) | `claude-sonnet-4-5` — resolved dated snapshot recorded in every `manifest.json` |
| Reader settings | `temperature=0`, `max_tokens=1024`, **no tools, no web access** |
| Judge model | `gpt-4o-2024-08-06` (upstream short name `gpt-4o`) at `temperature=0` |
| Dense embeddings | `text-embedding-3-small` |
| Lockfile | `uv.lock` (frozen) |

The reader is intentionally tool-free and web-free. The
`AnthropicReader` constructor asserts no `tools=` is passed and the API
call sends no tool definitions, no betas enabling browsing, nothing —
every system runs under the same fixed reader so accuracy differences
reflect retrieval/assembly, not model choice.

The judge (GPT-4o-as-judge) is also nondeterministic at the margin; we
pin it to a dated snapshot and run at temperature 0 to keep contribution
to ~±1 pt.

## Seven systems behind one interface

1. `full-context-oracle` — feed only the evidence sessions (upper bound).
2. `full-context-s` — stuff the entire history; truncates oldest-first if
   it exceeds the recorded `token_budget` (default 180k).
3. `rag-bm25` — BM25 retrieval, top-k.
4. `rag-dense` — `text-embedding-3-small` retrieval, top-k.
5. `activegraph-det-lexical` — deterministic ActiveGraph, **Mode A**. Turn-node
   graph built from raw text with co-occurrence + temporal edges (no LLM at
   ingest). Retrieves via IDF-weighted distinctive-token overlap and 1-hop
   temporal expansion, under a token budget that mirrors the turn-level RAG
   baselines (~2.5k context tokens by default).
6. `activegraph-det-embedding` — same graph + budget as (5), but retrieval
   relevance is cosine similarity against the pinned `text-embedding-3-small`
   model. Still NO LLM extraction at ingest; fully reproducible given the
   pinned embedding model.
7. `activegraph-memory-pack` — adapter for the external
   [`activegraph-memory`](https://github.com/yoheinakajima/activegraph-memory)
   pack. In v0.1 it keeps the reader context as conversation history only,
   while recording the pack's deterministic `memory_query -> retrieval_plan`,
   coverage report, confidence vector, and `memory_gateway` request shape in
   run metadata. Treat this as the Phase 1 integration/instrumentation cell,
   not a semantic-memory improvement claim yet.

All systems share the prompt template and reader settings. Token counts
(authoritative, from API `usage`) are logged for every query. The
ActiveGraph variants additionally guarantee **re-ingest equality**:
building the graph twice from the same instance produces a byte-identical
event log (asserted by `scripts/property_tests.py`).

## Setup

```bash
git clone <this-repo> && cd activegraph-longmemeval
make setup            # init submodule, install pinned deps from uv.lock
cp .env.example .env  # add ANTHROPIC_API_KEY and OPENAI_API_KEY
make data             # download datasets, write/verify data/CHECKSUMS.sha256
make smoke-ids        # build & commit config/smoke_ids.txt (one-time)
```

For the `activegraph-memory-pack` system, install or place the external
pack as a sibling checkout:

```bash
cd ..
git clone git@github.com:yoheinakajima/activegraph-memory.git activegraph-memory
cd activegraph-longmemeval
uv pip install -e ../activegraph-memory
```

If you are not using `uv`, install both repos into the same Python
environment:

```bash
python3.11 -m pip install -e ../activegraph-memory
```

If the package is not importable, `make tests` records the pack-adapter
check as an explicit skip and the normal benchmark systems still run.

`make data` writes `data/CHECKSUMS.sha256` on first run; commit it.
Subsequent runs verify and fail loudly on mismatch.

## Running

```bash
# single cell
make run SYSTEM=rag-bm25 DATA=s            # writes runs/<run_id>/
make eval RUN=runs/<run_id>                # invokes the frozen upstream judge

# offline property tests for baselines, ActiveGraph Mode A, and the
# activegraph-memory-pack Phase 1 adapter when the package is installed
# (no API required; embedding sub-variant skips without OPENAI_API_KEY).
make tests

# probe the resolved reader model snapshot (one-shot, needs ANTHROPIC_API_KEY)
make check-resolved-model

# four baselines only (skip the ActiveGraph variants), smoke-50, RAG × {turn, session}
make baselines-smoke   # needs ANTHROPIC_API_KEY + OPENAI_API_KEY

# full system matrix (4 baselines + ActiveGraph variants + activegraph-memory-pack)
make reproduce         # smoke = 50 frozen IDs from config/smoke_ids.txt
make reproduce-full    # every question in each dataset
```

`make reproduce` is the unit a reviewer re-runs by default (smoke, 50
questions). `make reproduce-full` runs all ~500 per dataset and enforces
`--require-authoritative-tokens` (the run fails if `context_tokens`
would be recorded with the char/4 fallback). Each cell writes
`runs/<run_id>/` with `hypotheses.jsonl`, `manifest.json`,
`scores.json`, and the upstream `eval.log`, then `paper/results_tables.md`
is regenerated with accuracy + mean tokens/query per cell.

`paper/BASELINE_SANITY.md` lists the expected qualitative orderings
across baselines; check it after every matrix run to catch silently
broken systems before ActiveGraph enters.

### Testing `activegraph-memory`

Use this ladder before spending on a full run:

```bash
# 1. Offline adapter contract
make tests

# 2. One smoke benchmark cell, after data is present
make run SYSTEM=activegraph-memory-pack DATA=s

# 3. Frozen 50-question matrix, after the adapter contract is green
make reproduce

# 4. Full 500 only if the 50-question run shows a reason to spend
make reproduce-full
```

The v0.1 pack adapter is expected to match the deterministic lexical
conversation context. The metrics to inspect first are metadata health
and answer-in-context sidecars, especially temporal and multi-session
questions. Accuracy lift should not be claimed until later pack behavior
actually changes retrieval or assembly with claims, supersession, temporal
refs, or evidence bundles.

## Reproducibility hooks

- `data/CHECKSUMS.sha256` — SHA-256 of each dataset file. First run records, later runs verify.
- `config/smoke_ids.txt` — frozen stratified 50-id smoke subset, seed=42. Committed.
- `uv.lock` — frozen full dep resolution. Committed.
- Submodule pinned to commit SHA above; `git submodule status` confirms.
- Per-run `manifest.json` captures: repo SHA, submodule SHA, dataset SHA-256,
  reader model **requested + resolved**, judge short name + resolved model, full
  config, seed, started/finished timestamps, wall-clock, per-question
  `{prompt_tokens, completion_tokens, context_tokens, truncated, elapsed_s}`,
  and the run-level `context_token_source` ∈ {`tiktoken`, `charfallback`}.
- tiktoken downloads its BPE on first use into the project-local
  `.tiktoken_cache/` (set automatically). Network access is required ONCE;
  after that all runs (including subprocess children) are offline-friendly.
  Paper runs assert `context_token_source == "tiktoken"` and fail otherwise.

## Notes for retrieval-recall extensions (future)

Upstream skips the 30 abstention instances when computing retrieval-recall
metrics (no ground-truth answer location). QA scoring via `evaluate_qa.py`
already handles `_abs` correctly, so this only matters if/when we add
retrieval-recall metrics — replicate the skip exactly when we do.

## Layout

```
config/run.yaml             # all knobs
config/smoke_ids.txt        # frozen 50-question subset (committed)
data/CHECKSUMS.sha256       # frozen dataset checksums (committed; files gitignored)
src/activegraph_lme/        # harness
  reader/                   # tool-free Anthropic reader
  systems/                  # system implementations behind a common interface
  activegraph/              # deterministic graph build + retrieval signals (Mode A)
  eval/run_judge.py         # wrapper around vendored evaluate_qa.py + print_qa_metrics.py
third_party/longmemeval/    # submodule, pinned commit
runs/<run_id>/              # hypotheses.jsonl, manifest.json, scores.json, eval.log
paper/results_tables.md     # autogenerated
```

## ActiveGraph deterministic (Mode A)

`src/activegraph_lme/systems/activegraph_det.py` implements both
sub-variants behind the same frozen adapter interface as every other
system:

```python
class ActiveGraphDetSystem:
    name: str   # "activegraph-det-lexical" | "activegraph-det-embedding"
    def ingest(self, instance) -> _State: ...
    def retrieve(self, state, question, question_date) -> AssembledContext: ...
```

The harness asserts on every run that `retrieve()` is deterministic
under a fixed state, and the offline property tests additionally assert
that `ingest()` is byte-identical when re-run on the same instance with
the same config (the re-ingest-equality property). Mode B
(LLM-extraction sub-variant) is gated on Mode A clearing the comparison
bar against the existing baselines.

## ActiveGraph Memory Pack Adapter

`src/activegraph_lme/systems/activegraph_memory_pack.py` is the benchmark
entry point for the external `activegraph-memory` repository. It imports
the installed package, or falls back to a sibling checkout named
`activegraph-memory`.

The adapter currently:

- builds the existing deterministic ActiveGraph lexical state;
- creates an `activegraph_memory.object_types.MemoryQuery`;
- calls `activegraph_memory.planner.plan_query`;
- records a `coverage_report`, confidence vector, and gateway-compatible
  retrieval request in `AssembledContext.meta`;
- leaves `AssembledContext.text` as conversation history only.

This keeps the published evaluation boundary clean while giving the new
pack a stable LongMemEval integration point for future semantic-memory
behavior.
