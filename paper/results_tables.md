# Results

Reader model (resolved): `claude-sonnet-4-5-20250929`

## Dataset: `oracle`

| system | granularity | overall acc | task-avg acc | abstain acc | mean ctx tok | mean prompt tok | mean compl tok | n truncated | n |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rag-bm25 | session | 0.866 | 0.879 | 0.967 | 5650 | 6499 | 102 | 0 | 500 |
| activegraph-det-lexical | n/a | 0.518 | 0.511 | 1.000 | 1540 | 1844 | 86 | 240 | 500 |
| rag-dense | session | 0.864 | 0.887 | 0.933 | 5650 | 6499 | 102 | 0 | 500 |
| full-context-oracle | n/a | 0.874 | 0.885 | 0.967 | 5650 | 6499 | 103 | 0 | 500 |
| activegraph-det-embedding | n/a | 0.868 | 0.889 | 0.933 | 2290 | 2688 | 94 | 438 | 500 |
| full-context-s | n/a | 0.866 | 0.883 | 0.933 | 5650 | 6499 | 104 | 0 | 500 |
| rag-bm25 | turn | 0.804 | 0.841 | 0.933 | 2703 | 3153 | 92 | 0 | 500 |
| rag-dense | turn | 0.848 | 0.875 | 0.900 | 2455 | 2876 | 94 | 0 | 500 |

### Per-type accuracy

| system | granularity | single-session-user | single-session-assistant | single-session-preference | multi-session | temporal-reasoning | knowledge-update |
|---|---|---:|---:|---:|---:|---:|---:|
| rag-bm25 | session | 0.971 | 1.000 | 0.800 | 0.804 | 0.827 | 0.872 |
| activegraph-det-lexical | n/a | 0.257 | 0.446 | 0.500 | 0.481 | 0.534 | 0.846 |
| rag-dense | session | 0.971 | 1.000 | 0.867 | 0.759 | 0.842 | 0.885 |
| full-context-oracle | n/a | 0.971 | 1.000 | 0.800 | 0.804 | 0.850 | 0.885 |
| activegraph-det-embedding | n/a | 0.957 | 1.000 | 0.833 | 0.782 | 0.812 | 0.949 |
| full-context-s | n/a | 0.957 | 1.000 | 0.833 | 0.767 | 0.857 | 0.885 |
| rag-bm25 | turn | 0.957 | 1.000 | 0.800 | 0.602 | 0.789 | 0.897 |
| rag-dense | turn | 0.957 | 1.000 | 0.833 | 0.729 | 0.804 | 0.923 |

## Dataset: `s`

| system | granularity | overall acc | task-avg acc | abstain acc | mean ctx tok | mean prompt tok | mean compl tok | n truncated | n |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rag-dense | session | 0.548 | 0.561 | 0.933 | 24133 | 27377 | 115 | 0 | 500 |
| activegraph-det-lexical | n/a | 0.618 | 0.648 | 1.000 | 2436 | 2848 | 86 | 491 | 500 |
| rag-bm25 | turn | 0.730 | 0.736 | 0.967 | 2516 | 2920 | 90 | 0 | 500 |
| full-context-s | n/a | 0.192 | 0.209 | 0.967 | 103743 | 117089 | 728 | 0 | 500 |
| activegraph-det-embedding | n/a | 0.856 | 0.879 | 0.967 | 2462 | 2881 | 94 | 500 | 500 |
| rag-bm25 | session | 0.560 | 0.571 | 0.967 | 27376 | 30897 | 118 | 0 | 500 |
| rag-dense | turn | 0.836 | 0.860 | 0.933 | 2366 | 2773 | 95 | 0 | 500 |
| full-context-oracle | n/a | 0.868 | 0.889 | 0.967 | 5650 | 6499 | 106 | 0 | 500 |

### Per-type accuracy

| system | granularity | single-session-user | single-session-assistant | single-session-preference | multi-session | temporal-reasoning | knowledge-update |
|---|---|---:|---:|---:|---:|---:|---:|
| rag-dense | session | 0.643 | 0.893 | 0.267 | 0.451 | 0.436 | 0.679 |
| activegraph-det-lexical | n/a | 0.771 | 0.857 | 0.467 | 0.406 | 0.564 | 0.821 |
| rag-bm25 | turn | 0.914 | 0.929 | 0.433 | 0.526 | 0.729 | 0.885 |
| full-context-s | n/a | 0.186 | 0.500 | 0.033 | 0.143 | 0.083 | 0.308 |
| activegraph-det-embedding | n/a | 0.957 | 1.000 | 0.833 | 0.774 | 0.797 | 0.910 |
| rag-bm25 | session | 0.671 | 0.893 | 0.233 | 0.338 | 0.556 | 0.731 |
| rag-dense | turn | 0.943 | 1.000 | 0.767 | 0.722 | 0.767 | 0.962 |
| full-context-oracle | n/a | 0.957 | 1.000 | 0.867 | 0.789 | 0.835 | 0.885 |

---
Each cell is a single run; the corresponding `runs/<run_id>/manifest.json` pins the repo SHA, submodule SHA, dataset SHA-256, exact reader/judge snapshots, seed, and per-query token counts.
