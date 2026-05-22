wrote /home/runner/workspace/paper/results_tables.md
5-20250929`

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
| full-context-s | n/a | 0.200 | 0.216 | 1.000 | 103257 | 116402 | 731 | 0 | 50 |
| rag-bm25 | session | 0.700 | 0.721 | 1.000 | 24928 | 28273 | 94 | 0 | 50 |
| rag-bm25 | turn | 0.800 | 0.835 | 1.000 | 2525 | 2955 | 70 | 0 | 50 |
| rag-dense | session | 0.660 | 0.728 | 1.000 | 23602 | 26848 | 77 | 0 | 50 |
| rag-dense | turn | 0.920 | 0.952 | 1.000 | 2398 | 2842 | 42 | 0 | 50 |

### Per-type accuracy

| system | granularity | single-session-user | single-session-assistant | single-session-preference | multi-session | temporal-reasoning | knowledge-update |
|---|---|---:|---:|---:|---:|---:|---:|
| full-context-oracle | n/a | 1.000 | 1.000 | 1.000 | 1.000 | 0.857 | 1.000 |
| full-context-s | n/a | 0.429 | 0.400 | 0.000 | 0.077 | 0.143 | 0.250 |
| rag-bm25 | session | 1.000 | 1.000 | 0.333 | 0.615 | 0.500 | 0.875 |
| rag-bm25 | turn | 1.000 | 1.000 | 0.667 | 0.769 | 0.571 | 1.000 |
| rag-dense | session | 0.857 | 1.000 | 0.667 | 0.538 | 0.429 | 0.875 |
| rag-dense | turn | 1.000 | 1.000 | 1.000 | 1.000 | 0.714 | 1.000 |

---
Each cell is a single run; the corresponding `runs/<run_id>/manifest.json` pins the repo SHA, submodule SHA, dataset SHA-256, exact reader/judge snapshots, seed, and per-query token counts.
