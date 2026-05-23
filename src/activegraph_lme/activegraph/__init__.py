"""Deterministic ActiveGraph (Mode A).

No LLM extraction. No reasoning at ingest. The graph is built from raw
LongMemEval turns via fixed, corpus-relative rules and re-ingest of the
same instance produces a byte-identical event log under a FrozenClock
seed. Retrieval scores Turn nodes via one of two pinned signals:

  * ``lexical``   — IDF-weighted distinctive-token overlap with the query
  * ``embedding`` — cosine similarity against ``text-embedding-3-small``

Both signals feed the same budgeted assembly so the lexical-vs-embedding
comparison is the only confound; the token budget mirrors the turn-level
RAG baselines (~2.5k context tokens).
"""

from .graph import Graph, Turn, build_graph
from .retrieve import assemble, score_lexical, score_embedding

__all__ = [
    "Graph",
    "Turn",
    "build_graph",
    "assemble",
    "score_lexical",
    "score_embedding",
]
