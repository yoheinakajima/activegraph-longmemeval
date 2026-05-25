"""Retrieval signals + budgeted assembly over the package-native graph.

Both signals (lexical IDF-overlap, embedding cosine) score Turn objects
that live in an ``activegraph.Graph``; the assembler:

  1. ranks turns by (score desc, sort_key asc — chronological tie-break),
  2. greedily adds turns under ``token_budget`` (computed via project tokenizer),
  3. for each selected turn, walks the 1-hop ``precedes`` neighborhood in
     the PACKAGE graph (via ``graph.neighborhood`` / ``graph.relations``)
     and pulls in the adjacent turn in the same session,
  4. emits selected turns in chronological order, joined by ``"\\n\\n"``.

Step 3 is the structural "graph use" that distinguishes this from
naive top-K — selecting an assistant reply pulls in the preceding user
turn (and vice versa) so the reader sees the local pair. It is
deterministic and free of tuned knobs. Walking happens through the
package's relation API; the dataclass turn list is only a chronological
view onto the same objects.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..tokens import count_tokens as _tok_count
from .graph import IngestState, Turn
from .stoplist import STOPLIST


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize_query(text: str, min_token_length: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for m in _TOKEN_RE.finditer(text):
        tok = m.group(0).lower()
        if len(tok) < min_token_length:
            continue
        if tok in STOPLIST:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


def _iter_turn_views(state: IngestState) -> list[Turn]:
    """Iterate Turn views in deterministic insertion order, but resolved
    through the PACKAGE graph (so this path actually queries ``objects()``
    rather than just reading the cache).
    """
    pkg_objs = state.graph.objects(type="Turn")
    # The package returns objects in insertion order; map back to our Turn
    # views via object_id so retrieve operates on the same projection the
    # event log produced.
    return [state.by_object_id[o.id] for o in pkg_objs]


# ---- lexical signal ----------------------------------------------------------


def score_lexical(state: IngestState, question: str, min_token_length: int) -> dict[str, float]:
    """IDF-weighted overlap between the question's distinctive tokens and
    each turn's distinctive tokens. Tokens outside the pruned vocab
    contribute 0.
    """
    q_tokens = _tokenize_query(question, min_token_length)
    turns = _iter_turn_views(state)
    n_turns = max(1, len(turns))
    idf: dict[str, float] = {}
    for tok in q_tokens:
        df = state.vocab_df.get(tok)
        if df is None:
            continue
        idf[tok] = math.log((n_turns + 1) / (df + 1)) + 1.0

    scores: dict[str, float] = {}
    for t in turns:
        toks = state.turn_tokens.get(t.turn_id, ())
        if not toks or not idf:
            scores[t.turn_id] = 0.0
            continue
        s = 0.0
        for tok in toks:
            w = idf.get(tok)
            if w is not None:
                s += w
        scores[t.turn_id] = s
    return scores


# ---- embedding signal --------------------------------------------------------


_EMBED_MAX_TOKENS = 8000  # safety margin under text-embedding-3-small's 8192 hard limit


def truncate_for_embedding(text: str) -> str:
    """Truncate to <=_EMBED_MAX_TOKENS cl100k tokens (the embedding model's hard
    input limit). Deterministic; affects ONLY the similarity vector, never the
    text assembled into the reader's context. Shared by rag-dense and
    activegraph-det-embedding so both embed long inputs identically."""
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    toks = enc.encode(text, disallowed_special=())
    if len(toks) <= _EMBED_MAX_TOKENS:
        return text
    return enc.decode(toks[:_EMBED_MAX_TOKENS])


@dataclass
class EmbeddingClient:
    model: str
    _client: Any | None = None
    # Content-addressed cache so re-embedding the same text is byte-identical
    # for the harness's repeat-call determinism check.
    _cache: dict[str, np.ndarray] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._cache is None:
            self._cache = {}

    def _ensure(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        return self._client

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1536), dtype=np.float32)
        to_fetch: list[tuple[int, str]] = []
        for i, t in enumerate(texts):
            if t not in self._cache:
                to_fetch.append((i, t))

        if to_fetch:
            cli = self._ensure()
            new_texts = [t for _, t in to_fetch]
            B = 96
            new_vecs: list[np.ndarray] = []
            for i in range(0, len(new_texts), B):
                _batch = [truncate_for_embedding(t) for t in new_texts[i : i + B]]
                resp = cli.embeddings.create(model=self.model, input=_batch)
                new_vecs.extend(np.asarray(d.embedding, dtype=np.float32) for d in resp.data)
            for (_, t), v in zip(to_fetch, new_vecs):
                n = float(np.linalg.norm(v))
                self._cache[t] = v / n if n > 0 else v

        return np.stack([self._cache[t] for t in texts], axis=0)


def score_embedding(
    state: IngestState,
    question: str,
    embedder: EmbeddingClient,
    turn_embeddings: np.ndarray | None = None,
) -> tuple[dict[str, float], np.ndarray]:
    """Cosine similarity between the question and each turn (L2-normalized).

    ``turn_embeddings`` may be passed in pre-computed (cached); otherwise it is
    computed once and returned for reuse.
    """
    turns = _iter_turn_views(state)
    if turn_embeddings is None:
        turn_embeddings = embedder.embed([t.text for t in turns])
    if not turns:
        return {}, turn_embeddings
    q_vec = embedder.embed([question])[0]
    raw = turn_embeddings @ q_vec
    sims: dict[str, float] = {}
    for i, t in enumerate(turns):
        sims[t.turn_id] = float(raw[i])
    return sims, turn_embeddings


# ---- assembly ---------------------------------------------------------------


@dataclass
class AssemblyResult:
    text: str
    truncated: bool
    selected_turn_ids: list[str]
    n_seeds: int
    n_expanded: int


def _temporal_neighbors_via_graph(state: IngestState, turn: Turn) -> list[Turn]:
    """1-hop temporal neighbors of ``turn`` as projected by the package's
    ``neighborhood``: walk relations of type ``precedes`` only and return
    the adjacent Turn views in deterministic order.
    """
    nbr_objs, nbr_rels = state.graph.neighborhood(turn.object_id, depth=1)
    neighbors: list[Turn] = []
    seen: set[str] = set()
    for r in nbr_rels:
        if r.type != "precedes":
            continue
        other = r.target if r.source == turn.object_id else r.source
        if other == turn.object_id or other in seen:
            continue
        seen.add(other)
        v = state.by_object_id.get(other)
        if v is not None:
            neighbors.append(v)
    return neighbors


def assemble(
    state: IngestState,
    scores: dict[str, float],
    *,
    token_budget: int,
) -> AssemblyResult:
    turns = _iter_turn_views(state)
    if not turns:
        return AssemblyResult(text="", truncated=False, selected_turn_ids=[], n_seeds=0, n_expanded=0)

    by_id: dict[str, Turn] = {t.turn_id: t for t in turns}
    sort_key = {t.turn_id: (t.session_date, t.session_idx, t.turn_idx) for t in turns}

    ranked = sorted(
        (t.turn_id for t in turns),
        key=lambda tid: (-scores.get(tid, 0.0), sort_key[tid]),
    )

    selected: list[str] = []
    selected_set: set[str] = set()
    running = 0
    truncated = False

    def _fits(tid: str) -> bool:
        n = _tok_count(by_id[tid].text) + 2  # +2 ≈ "\n\n" joiner
        return running + n <= token_budget

    def _add(tid: str) -> bool:
        nonlocal running
        if tid in selected_set:
            return True
        n = _tok_count(by_id[tid].text) + 2
        if running + n > token_budget:
            return False
        selected.append(tid)
        selected_set.add(tid)
        running += n
        return True

    # 1) Seed selection by score.
    n_seeds = 0
    for tid in ranked:
        if scores.get(tid, 0.0) <= 0.0 and n_seeds > 0:
            break
        if not _fits(tid):
            truncated = True
            continue
        if _add(tid):
            n_seeds += 1

    # 2) 1-hop temporal expansion via the PACKAGE graph's neighborhood.
    n_expanded = 0
    expansion_targets: list[str] = []
    seeds_snapshot = list(selected)
    for tid in seeds_snapshot:
        t = by_id[tid]
        for neigh in _temporal_neighbors_via_graph(state, t):
            if neigh.turn_id in selected_set:
                continue
            expansion_targets.append(neigh.turn_id)

    expansion_targets = sorted(set(expansion_targets), key=lambda tid: sort_key[tid])
    for tid in expansion_targets:
        if not _fits(tid):
            truncated = True
            continue
        if _add(tid):
            n_expanded += 1

    # 3) Emit in chronological order.
    selected.sort(key=lambda tid: sort_key[tid])
    text = "\n\n".join(by_id[tid].text for tid in selected)
    return AssemblyResult(
        text=text,
        truncated=truncated,
        selected_turn_ids=selected,
        n_seeds=n_seeds,
        n_expanded=n_expanded,
    )
