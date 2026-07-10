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
   pack. The v0.3 adapter compiles canonical entities/events, state histories,
   preferences, quantities, temporal refs, and list positions; runs fielded
   embeddings plus graph propagation; executes typed query operators; assesses
   sufficiency/conflicts; and prepends calibrated proof-oriented evidence to
   source conversation history.

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
# activegraph-memory-pack adapter when the package is installed
# (no API required; embedding paths skip without OPENAI_API_KEY).
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

# 4. Full 500 with durable resume/caching
python3.11 -m activegraph_lme.cli run \
  --system activegraph-memory-pack \
  --dataset s \
  --config config/run.activegraph-memory-v2.yaml \
  --run-id agmem-v2-full-YYYYMMDDTHHMMSSZ \
  --resume \
  --require-authoritative-tokens
```

The current pack adapter compiles role-aware extracted claims from
`data/sem_extract_cache/seed-A-v2.jsonl` into the external
`activegraph-memory` runtime, then retrieves evidence bundles that render
memory-claim headers directly above their source turns. The v2 config selects
the `max_quality` profile with a 10,000-token context budget, three targeted
retrieval rounds, entity/event embeddings, and persistent compiled-vector
caching. Optional LLM reasoning is not attached in this benchmark cell, so the
profile remains deterministic. If the extraction cache is missing, offline
tests fall back to deterministic turn-derived claims rather than calling an
extractor.

Every completed query writes exact retrieval context and metadata to
`retrieval_records.jsonl`. `query_records.jsonl` includes retrieval latency,
estimated retrieval cost, profile, and proof completion. `manifest.json`
aggregates mean/p95 retrieval latency, total estimated retrieval cost, and
proof-complete rate.

### ActiveGraph Memory result reports

- [`ACTIVEGRAPH_MEMORY_FULL_RUN_REPORT.md`](ACTIVEGRAPH_MEMORY_FULL_RUN_REPORT.md)
  documents the historical pre-v0.2 `0.878` full run and the failures that
  motivated typed aggregates, temporal execution, state history, preferences,
  and a larger context budget.
- [`ACTIVEGRAPH_MEMORY_V2_REPORT.md`](ACTIVEGRAPH_MEMORY_V2_REPORT.md) documents
  the complete follow-on experiment sequence, current v2 architecture,
  speed/cost telemetry, retrieval-hit analysis, the latest `0.824` regression,
  and the changes that motivated v3.
- [`ACTIVEGRAPH_MEMORY_V3_REPORT.md`](ACTIVEGRAPH_MEMORY_V3_REPORT.md) documents
  the completed 0.3 architecture, 0.98 smoke, 0.832 full-500 result, retrieval
  and reader decomposition, proof/sufficiency calibration, operator diagnostics,
  speed/cost telemetry, and remaining general memory-system gaps.
- [`ACTIVEGRAPH_MEMORY_V4_REPORT.md`](ACTIVEGRAPH_MEMORY_V4_REPORT.md) records
  the five-experiment evidence-quality bundle, the one-smoke/one-full policy,
  offline controls, and v4 benchmark results.

Latest typed v3 run:

```text
runs/agmem-v3-full-20260710T071914Z__activegraph-memory-pack__s__full
overall accuracy:       0.832
task-averaged accuracy: 0.8508
abstention accuracy:    0.9667
```

The run used `activegraph-memory` 0.3.0 commit `58f235b`, `max_quality`, a
10,000-token budget, fielded `text-embedding-3-small` retrieval, adaptive
sufficiency assessment, and calibrated candidate rendering. Exact gold-turn
recall was 89.79% on non-abstention questions. The fixed reader still missed 57
questions despite every exact gold turn being present. Optional LLM reasoning
was not attached, so reasoning-stage cost remained zero.

Previous typed v2 proof run:

```text
runs/agmem-v2-proof-full-20260710T015130Z__activegraph-memory-pack__s__full
overall accuracy:       0.824
task-averaged accuracy: 0.8515
abstention accuracy:    0.9333
```

The run used `activegraph-memory` commit `71e6683`, `max_quality`, a
10,000-token budget, and fielded `text-embedding-3-small` retrieval. It did not
attach an optional LLM reasoning backend, so the classification, strategy,
analysis, and packaging stages remained deterministic. Exact gold source turns
were present for 90.43% of non-abstention questions, but 63 answers were wrong
despite an exact gold turn being present. The post-run reader contract now
treats proof completion as structural evidence coverage, not semantic answer
verification. The v3 run above measures that correction and reduces these
reader-with-exact-evidence failures from 63 to 57, while exact-turn recall falls
slightly from 90.43% to 89.79%.

Historical pre-v0.2 smoke result for the first compiled-memory adapter:

```text
runs/agmem-fullarch2-smoke-20260709T014835Z__activegraph-memory-pack__s__smoke
overall accuracy:       0.94
task-averaged accuracy: 0.9634
abstention accuracy:    1.0
```

Per-type accuracy on that 50-question smoke:

```text
single-session-user        1.0000
single-session-preference  1.0000
single-session-assistant   1.0000
multi-session              0.9231
temporal-reasoning         0.8571
knowledge-update           1.0000
```

Historical pre-v0.2 full-500 result:

```text
runs/agmem-fullarch-full-20260709T022124Z__activegraph-memory-pack__s__full
overall accuracy:       0.878
task-averaged accuracy: 0.8943
abstention accuracy:    0.9667
```

Per-type accuracy on the full `longmemeval_s_cleaned.json` run:

```text
single-session-user        0.9571  (n=70)
single-session-preference  0.8333  (n=30)
single-session-assistant   0.9821  (n=56)
multi-session              0.7895  (n=133)
temporal-reasoning         0.8421  (n=133)
knowledge-update           0.9615  (n=78)
```

This is a small full-set gain over the strongest local `s` full baselines
recorded in `runs/`: `rag-dense` turn-level at `0.836`, deterministic
ActiveGraph embedding at `0.85`, and `activegraph-sem-hybrid` at `0.876`
overall. The smoke score was optimistic; the full-set drag is concentrated
in multi-session and preference questions.

Full cells now write enough state to be resumed safely:

- `query_records.jsonl` is appended and fsynced after every completed
  question; `--resume` skips those question ids.
- `run_events.jsonl` records `run.started`, `run.resumed`,
  `query.started`, `query.completed`, `run.failed`, and `run.completed`
  events for audit/recovery.
- `run_state.json` and `manifest.partial.json` are rewritten after each
  question; final `manifest.json` is stamped at completion.
- `.embedding_cache/embeddings.sqlite3` stores normalized embedding
  vectors keyed by `(model, sha256(truncated_model_input))`, so crashes
  and reruns do not re-pay already computed embedding calls. Set
  `AGLME_EMBEDDING_CACHE=/path/to/embeddings.sqlite3` to share the cache
  across checkouts, or `AGLME_EMBEDDING_CACHE=off` to disable it.
- `.embedding_cache/activegraph-memory-v2.sqlite3` stores fielded compiled
  vectors using the pack's model/field/subject/text-hash key. This cache makes
  interrupted v0.2 runs restart without re-embedding unchanged corpus rows.

## Reproducibility hooks

- `data/CHECKSUMS.sha256` — SHA-256 of each dataset file. First run records, later runs verify.
- `config/smoke_ids.txt` — frozen stratified 50-id smoke subset, seed=42. Committed.
- `uv.lock` — frozen full dep resolution. Committed.
- Submodule pinned to commit SHA above; `git submodule status` confirms.
- Per-run `manifest.json` captures: repo SHA, submodule SHA, dataset SHA-256,
  reader model **requested + resolved**, judge short name + resolved model, full
  config, seed, started/finished timestamps, wall-clock, per-question
  `{prompt_tokens, completion_tokens, context_tokens, truncated, elapsed_s,
  retrieval_latency_ms, retrieval_cost_usd, runtime_profile, proof_complete}`,
  run-level `context_token_source` in {`tiktoken`, `charfallback`}, and
  embedding-cache statistics when a system uses embeddings.
- Per-run `query_records.jsonl`, `run_events.jsonl`, and `run_state.json`
  make API-backed cells resumable with `--resume`.
- Per-run `retrieval_records.jsonl` preserves the exact reader context, query
  IR, compiled proof, selected IDs, and stage telemetry for miss analysis.
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
  query_records.jsonl       # per-question durable provenance for resume
  run_events.jsonl          # append-only benchmark event stream
  run_state.json            # latest resume pointer
paper/results_tables.md     # autogenerated
.embedding_cache/           # local durable embedding cache (gitignored)
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
- compiles role-aware memory claims from the frozen extraction cache or a
  deterministic turn-derived fallback;
- runs the selected `MemoryRuntimeProfile` through query analysis, fielded
  lexical/embedding retrieval, graph signal propagation, typed execution, and
  provenance-preserving packaging;
- persists compiled corpus vectors through `SQLiteEmbeddingStore`;
- records the `retrieval_plan`, `coverage_report`, confidence vector,
  evidence bundle, compiled proof, query IR, per-stage telemetry, selected
  turn/claim/event ids, and gateway-compatible retrieval request;
- renders the compiled proof packet before source conversation turns passed to
  the fixed reader.

This keeps the published evaluation boundary clean: the fixed reader and judge
do not change, while retrieval semantics, context size, stage usage, and cost
remain explicit in config and run artifacts.
