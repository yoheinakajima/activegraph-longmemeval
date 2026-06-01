# Stage-1 Full-500 Results

## QA Accuracy
- det-embedding: 0.850 (425/500)
- sem-hybrid v2: 0.876 (438/500)
- Aggregate McNemar: +13, p=0.1175 (not significant)
- Temporal-reasoning McNemar: +11/127, p=0.0266 (significant)

## Retrieval AIC (turn-level)
- det-embedding: 86.2% (405/470)
- sem-hybrid v2: 90.0% (423/470)
- Aggregate McNemar: +18, p=0.030 (significant)

### Per-category AIC McNemar
| Category | n | DET | HYB | b-c | p |
|---|---:|---:|---:|---:|---:|
| multi-session | 121 | 76.0% | 86.8% | +13 | 0.0072 ** |
| temporal-reasoning | 127 | 79.5% | 90.6% | +14 | 0.0026 ** |
| single-session-assistant | 56 | 100.0% | 92.9% | -4 | 0.125 |
| single-session-preference | 30 | 80.0% | 70.0% | -3 | 0.453 |
| single-session-user | 64 | 98.4% | 96.9% | -1 | 1.000 |
| knowledge-update | 72 | 95.8% | 94.4% | -1 | 1.000 |

## Reader-failed-with-evidence
- det-embedding: 37
- sem-hybrid v2: 37
- Reader bottleneck identical across systems.
