# Answer-in-context results (LongMemEval-S)

Companion artifact for the benchmark note *"Not a Memory System Yet: ActiveGraph's
LongMemEval-S Result."* This file records the **answer-in-context** measurement that the
post's "Still missing → Retrieval metrics" item asked for.

## What this measures

Answer-in-context asks a narrower question than end-to-end QA: **did the gold-evidence
turns/sessions actually land in the context handed to the reader?** It is computed by
replaying *retrieval only* (no reader, no LLM call beyond query embedding) over an
already-completed run, then checking the assembled context against LongMemEval's
`has_answer` evidence labels.

- `turn-AIC` — fraction of questions where the gold **turn** IDs are a subset of the
  retrieved turn IDs.
- `sess-AIC` — fraction where the gold **session** IDs are a subset of the retrieved
  session IDs.
- `miss_turn` / `miss_sess` — count of questions where turn-/session-level evidence was
  not retrieved.
- `rfwe` (reader-failed-with-evidence) — count where evidence **was** retrieved but the
  answer was still judged wrong.
- `acc` — per-type accuracy recomputed from `autoeval_label` (sanity check vs. the paper).

The 30 abstention (`_abs`) questions have no gold evidence location and are excluded, as
upstream does for retrieval-recall metrics, leaving **n = 470**.

## Provenance

- Matrix: `runs/matrix_20260524T050742Z.json` (the same matrix behind the paper's main
  table; ActiveGraph-det-embedding = 85.6% / 500).
- Cells replayed: `activegraph-det-embedding__s` and `rag-dense__s__g-turn`.
- Method: `scripts/aic_sidecar.py` replays retrieval and **asserts the reconstructed
  context is byte-identical to the published run's** `system.retrieve().text`, and that a
  repeat call is identical, so these numbers correspond to the published cells.
- Scoring: `scripts/answer_in_context.py`.
- Turn-ID scheme: harness-native `{session_id}#{turn_idx}` (the dataset's haystack turns
  do not carry upstream turn-IDs; gold and retrieved IDs are synthesized the same way).
- Reader/judge unchanged from the main run: reader `claude-sonnet-4-5` (temp 0, no tools);
  judge `gpt-4o-2024-08-06` (temp 0).

Recomputed accuracy here (AG 0.849, dense 0.830) matches the paper's QA accuracy minus the
excluded abstention questions (0.856 / 0.836 over 500), confirming the replay is faithful.

## Overall (n = 470)

| System | turn-AIC | sess-AIC | acc | rfwe |
|---|---|---|---|---|
| activegraph-det-embedding | **0.862** | **0.949** | 0.849 | 35 |
| rag-dense (turn) | 0.817 | 0.930 | 0.830 | 27 |

ActiveGraph places the gold evidence turn in context on 86.2% of questions vs. dense
turn-RAG's 81.7% — 21 fewer turn-level misses across 470. The accuracy edge (+1.9) tracks
this retrieval edge rather than reader variance. ActiveGraph does, however, have slightly
**more** reader-failures-with-evidence (35 vs. 27): it retrieves enough additional evidence
to win on accuracy despite a marginally higher reader-fumble rate. The two effects are the
same order of magnitude: +21 net turn-AIC hits (25 won, 4 lost) against +8
reader-failures-with-evidence (35 vs. 27). The significant retrieval advantage is partly
absorbed by reader-fumbles on the extra context it surfaces, which is why end-to-end QA
lands at +1.9 rather than tracking the full retrieval gap. The discordance is one-directional: of the 29 questions where the systems disagree on
turn-AIC, ActiveGraph wins 25 and loses 4 (exact McNemar, p = 0.0001), so the +4.5-point
retrieval edge is statistically established rather than a point estimate. This is a retrieval
*outcome*, not evidence of graph *causality* — isolating the latter still requires the
ablations listed in the post.

## Per-type — ActiveGraph-det-embedding

| type | n | turn-AIC | sess-AIC | miss_turn | miss_sess | rfwe | acc |
|---|---|---|---|---|---|---|---|
| knowledge-update | 72 | 0.958 | 0.986 | 3 | 1 | 5 | 0.903 |
| multi-session | 121 | 0.760 | 0.959 | 29 | 5 | 14 | 0.752 |
| single-session-assistant | 56 | 1.000 | 1.000 | 0 | 0 | 0 | 1.000 |
| single-session-preference | 30 | 0.800 | 1.000 | 6 | 0 | 4 | 0.833 |
| single-session-user | 64 | 0.984 | 1.000 | 1 | 0 | 2 | 0.953 |
| temporal-reasoning | 127 | 0.795 | 0.858 | 26 | 18 | 10 | 0.795 |
| **overall** | **470** | **0.862** | **0.949** | **65** | **24** | **35** | **0.849** |

## Per-type — dense turn-RAG

| type | n | turn-AIC | sess-AIC | miss_turn | miss_sess | rfwe | acc |
|---|---|---|---|---|---|---|---|
| knowledge-update | 72 | 0.944 | 1.000 | 4 | 0 | 2 | 0.958 |
| multi-session | 121 | 0.678 | 0.926 | 39 | 9 | 8 | 0.703 |
| single-session-assistant | 56 | 1.000 | 1.000 | 0 | 0 | 0 | 1.000 |
| single-session-preference | 30 | 0.767 | 1.000 | 7 | 0 | 5 | 0.767 |
| single-session-user | 64 | 0.984 | 1.000 | 1 | 0 | 3 | 0.938 |
| temporal-reasoning | 127 | 0.724 | 0.811 | 35 | 24 | 9 | 0.764 |
| **overall** | **470** | **0.817** | **0.930** | **86** | **33** | **27** | **0.830** |

## Reading the per-type results

**Knowledge-update is reader reconciliation, not retrieval.** Both systems retrieve the
gold evidence at ~95% turn / 98–100% session. ActiveGraph retrieves it marginally *better*
than dense-RAG (95.8% vs. 94.4% turn) yet scores *lower* (90.3% vs. 95.8%), because its
assembled context produces more reader-failures-with-evidence (5 vs. 2 of ~72). The
post's earlier speculation that a semantic layer helps here by avoiding "rediscovery from
prose" is not supported — the evidence is already in context. The semantic layer's value on
knowledge-update is in **marking facts as superseded before assembly** to ease reader
reconciliation, not in improving retrieval recall, which is near-ceiling.

**Multi-session and temporal show a session-vs-turn gap.** Both systems retrieve the right
*session* far more often than the right *turn* (ActiveGraph multi-session: 95.9% session
vs. 76.0% turn), and accuracy tracks the turn number. These are the categories with real
turn-level retrieval headroom, where ActiveGraph's advantage is largest (multi-session
+8.2, temporal +7.1 turn-AIC) and where it most directly converts to accuracy. This is
directionally consistent with the RAG-vs-GraphRAG literature (graph structure helps on
multi-hop), but is **not** proof of causality.

## What the benchmark cannot measure

The substrate's retrieval ceiling is already high exactly where semantic memory is meant to
help — knowledge-update sits at ~95% turn / ~99% session recall for both systems — so the
bottleneck there is reader reconciliation, not retrieval. Whether typed superseded-fact
representation actually eases that reconciliation is precisely what LongMemEval-S cannot
measure, and is the real open question for the next experiment.

## Caveat

This measures whether **labeled** gold evidence reached the reader, not whether
**sufficient** evidence did. A question answerable from context the labelers did not mark
would register as a spurious reader-failure-with-evidence. This is the standard limitation
of `has_answer`-based evidence metrics and applies equally to all systems here.

## Reproduce

```
uv run python scripts/aic_sidecar.py        runs/<activegraph-det-embedding s cell>
uv run python scripts/answer_in_context.py  runs/<activegraph-det-embedding s cell>
uv run python scripts/aic_sidecar.py        runs/<rag-dense s g-turn cell>
uv run python scripts/answer_in_context.py  runs/<rag-dense s g-turn cell>
```

The sidecar requires the local dataset and run directories (`make data` regenerates the
dataset, verified against `data/CHECKSUMS.sha256`). Embedding-mode replay needs
`OPENAI_API_KEY` for query embedding; no Anthropic/reader call is made.
