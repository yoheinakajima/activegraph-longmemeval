# Results

Reader model (resolved): `claude-sonnet-4-5-20250929`

## Dataset: `oracle`

| system | granularity | overall acc | task-avg acc | abstain acc | mean ctx tok | mean prompt tok | mean compl tok | n truncated | n |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full-context-oracle | n/a | 0.920 | 0.951 | 1.000 | 5025 | 5780 | 79 | 0 | 50 |
| full-context-s | n/a | 0.900 | 0.939 | 1.000 | 5030 | 5790 | 80 | 0 | 50 |
| rag-bm25 | session | 0.940 | 0.963 | 1.000 | 5025 | 5780 | 79 | 0 | 50 |
| rag-bm25 | turn | 0.880 | 0.926 | 1.000 | 2960 | 3436 | 72 | 0 | 50 |
| rag-dense | session | 0.880 | 0.926 | 1.000 | 5025 | 5780 | 80 | 0 | 50 |
| rag-dense | turn | 0.920 | 0.952 | 1.000 | 2665 | 3135 | 82 | 0 | 50 |

### Per-type accuracy

| system | granularity | single-session-user | single-session-assistant | single-session-preference | multi-session | temporal-reasoning | knowledge-update |
|---|---|---:|---:|---:|---:|---:|---:|
| full-context-oracle | n/a | 1.000 | 1.000 | 1.000 | 0.846 | 0.857 | 1.000 |
| full-context-s | n/a | 1.000 | 1.000 | 1.000 | 0.846 | 0.786 | 1.000 |
| rag-bm25 | session | 1.000 | 1.000 | 1.000 | 0.846 | 0.929 | 1.000 |
| rag-bm25 | turn | 1.000 | 1.000 | 1.000 | 0.769 | 0.786 | 1.000 |
| rag-dense | session | 1.000 | 1.000 | 1.000 | 0.769 | 0.786 | 1.000 |
| rag-dense | turn | 1.000 | 1.000 | 1.000 | 0.923 | 0.786 | 1.000 |

## Dataset: `s`

| system | granularity | overall acc | task-avg acc | abstain acc | mean ctx tok | mean prompt tok | mean compl tok | n truncated | n |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full-context-oracle | n/a | 0.960 | 0.976 | 1.000 | 4611 | 5279 | 70 | 0 | 50 |

### Per-type accuracy

| system | granularity | single-session-user | single-session-assistant | single-session-preference | multi-session | temporal-reasoning | knowledge-update |
|---|---|---:|---:|---:|---:|---:|---:|
| full-context-oracle | n/a | 1.000 | 1.000 | 1.000 | 1.000 | 0.857 | 1.000 |

---
Each cell is a single run; the corresponding `runs/<run_id>/manifest.json` pins the repo SHA, submodule SHA, dataset SHA-256, exact reader/judge snapshots, seed, and per-query token counts.
