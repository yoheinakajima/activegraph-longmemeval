"""Deterministic ActiveGraph (Mode A) — package-native.

Built on top of the real ``activegraph`` pip package: every Turn / Session
is an ``activegraph.Object``, every adjacency or co-occurrence is an
``activegraph.Relation``, and the event log is ``activegraph.Graph.events``.
Re-ingest of the same instance produces a byte-identical event log under
a FrozenClock + deterministic run_id. Retrieval scores Turn objects via
one of two pinned signals:

  * ``lexical``   — IDF-weighted distinctive-token overlap with the query
  * ``embedding`` — cosine similarity against ``text-embedding-3-small``

Both signals feed the same budgeted assembly so the lexical-vs-embedding
comparison is the only confound; the token budget mirrors the turn-level
RAG baselines (~2.5k context tokens).
"""

from .graph import IngestState, Turn, build_graph
from .retrieve import assemble, score_embedding, score_lexical

__all__ = [
    "IngestState",
    "Turn",
    "build_graph",
    "assemble",
    "score_lexical",
    "score_embedding",
]
