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

## Five systems behind one interface

1. `full-context-oracle` — feed only the evidence sessions (upper bound).
2. `full-context-s` — stuff the entire history; truncates oldest-first if
   it exceeds the recorded `token_budget` (default 180k).
3. `rag-bm25` — BM25 retrieval, top-k.
4. `rag-dense` — `text-embedding-3-small` retrieval, top-k.
5. `activegraph` — **STUB this round** (recency under budget). The
   `ingest()`/`retrieve()` interface is frozen; round two replaces the
   internals without touching anything else.

All five share the prompt template and reader settings. Token counts
(authoritative, from API `usage`) are logged for every query.

## Setup

```bash
git clone <this-repo> && cd activegraph-longmemeval
make setup            # init submodule, install pinned deps from uv.lock
cp .env.example .env  # add ANTHROPIC_API_KEY and OPENAI_API_KEY
make data             # download datasets, write/verify data/CHECKSUMS.sha256
make smoke-ids        # build & commit config/smoke_ids.txt (one-time)
```

`make data` writes `data/CHECKSUMS.sha256` on first run; commit it.
Subsequent runs verify and fail loudly on mismatch.

## Running

```bash
# single cell
make run SYSTEM=rag-bm25 DATA=s            # writes runs/<run_id>/
make eval RUN=runs/<run_id>                # invokes the frozen upstream judge

# offline property tests for the four baselines (no API)
make tests

# probe the resolved reader model snapshot (one-shot, needs ANTHROPIC_API_KEY)
make check-resolved-model

# four baselines only (skip the activegraph stub), smoke-50, RAG × {turn, session}
make baselines-smoke   # needs ANTHROPIC_API_KEY + OPENAI_API_KEY

# full 5-system matrix
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
  systems/                  # five system implementations behind a common interface
  eval/run_judge.py         # wrapper around vendored evaluate_qa.py + print_qa_metrics.py
third_party/longmemeval/    # submodule, pinned commit
runs/<run_id>/              # hypotheses.jsonl, manifest.json, scores.json, eval.log
paper/results_tables.md     # autogenerated
```

## Round two: real ActiveGraph

`src/activegraph_lme/systems/activegraph_stub.py` defines the frozen
adapter interface:

```python
class ActiveGraphSystem:
    def ingest(self, instance) -> GraphState: ...
    def retrieve(self, state, question, question_date) -> AssembledContext: ...
```

The harness asserts on every run that `retrieve()` is deterministic
under a fixed state. Replace the body of these two methods; do not
change their signatures or what they return.
