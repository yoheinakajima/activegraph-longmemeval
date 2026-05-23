"""Retrieval signals + budgeted assembly.

Two pinned signals share the same selection/assembly path. The signal
returns a per-Turn score; the assembler walks the score-ranked seeds in
order and, for each seed, also pulls in its temporal neighbors before
considering the next seed. Temporal neighbors are taken from two
graph-derived relations:

  * the intra-session 1-hop pair (existing structural use: pulling in
    the paired user/assistant turn alongside a relevant utterance), and
  * the global session-date ordering (new): a seed pulls in the turns
    immediately before/after it in the chronological ordering across
    all sessions, within ``temporal_expansion_hops``. This is what
    survives consecutive-day evidence on temporal-reasoning questions
    against unrelated higher-similarity turns.

The interleave (seed -> its expansions -> next seed) is the budget
discipline that preserves a relevant turn's temporal context against
lower-scoring competitors. Final turns are emitted in chronological
order joined by ``"\\n\\n"``. Deterministic and free of tuned knobs.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..tokens import count_tokens as _tok_count
from .graph import Graph
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


# ---- lexical signal ----------------------------------------------------------

def score_lexical(graph: Graph, question: str, min_token_length: int) -> dict[str, float]:
    """IDF-weighted overlap between the question's distinctive tokens and each
    turn's distinctive tokens. Tokens outside the pruned vocab contribute 0.
    """
    q_tokens = _tokenize_query(question, min_token_length)
    # Build per-token IDF using DF of the pruned vocab; tokens not in vocab → 0.
    n_turns = max(1, len(graph.turns))
    idf: dict[str, float] = {}
    for tok in q_tokens:
        df = graph.vocab_df.get(tok)
        if df is None:
            continue
        idf[tok] = math.log((n_turns + 1) / (df + 1)) + 1.0  # smoothed, always >0

    scores: dict[str, float] = {}
    for t in graph.turns:
        toks = graph.turn_tokens.get(t.turn_id, ())
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

@dataclass
class EmbeddingClient:
    model: str
    _client: Any | None = None
    # Content-addressed cache so re-embedding the same text (e.g. the question
    # on the harness's repeat-call determinism check) is byte-identical. The
    # cache is per-process and never persisted; it costs at most a few MB for
    # a 50-question smoke run.
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
        # Partition into (cached) and (to_embed) preserving order so the
        # output rows align with the input.
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
                resp = cli.embeddings.create(model=self.model, input=new_texts[i : i + B])
                new_vecs.extend(np.asarray(d.embedding, dtype=np.float32) for d in resp.data)
            for (_, t), v in zip(to_fetch, new_vecs):
                # L2-normalize on insert so callers always read unit vectors.
                n = float(np.linalg.norm(v))
                self._cache[t] = v / n if n > 0 else v

        return np.stack([self._cache[t] for t in texts], axis=0)


def score_embedding(
    graph: Graph,
    question: str,
    embedder: EmbeddingClient,
    turn_embeddings: np.ndarray | None = None,
) -> tuple[dict[str, float], np.ndarray]:
    """Cosine similarity between the question and each turn (L2-normalized).

    ``turn_embeddings`` may be passed in pre-computed (cached); otherwise it is
    computed once and returned for reuse.
    """
    if turn_embeddings is None:
        turn_embeddings = embedder.embed([t.text for t in graph.turns])
    q_vec = embedder.embed([question])[0] if graph.turns else np.zeros(1, dtype=np.float32)
    sims: dict[str, float] = {}
    if not graph.turns:
        return sims, turn_embeddings
    raw = turn_embeddings @ q_vec
    for i, t in enumerate(graph.turns):
        sims[t.turn_id] = float(raw[i])
    return sims, turn_embeddings


# ---- assembly ---------------------------------------------------------------

@dataclass
class AssemblyResult:
    text: str
    truncated: bool
    selected_turn_ids: list[str]
    n_seeds: int                 # turns selected directly by score
    n_expanded: int              # turns added via 1-hop temporal expansion


def assemble(
    graph: Graph,
    scores: dict[str, float],
    *,
    token_budget: int,
    temporal_expansion_hops: int = 1,
) -> AssemblyResult:
    if not graph.turns:
        return AssemblyResult(text="", truncated=False, selected_turn_ids=[], n_seeds=0, n_expanded=0)

    by_id = {t.turn_id: t for t in graph.turns}
    sort_key = {t.turn_id: (t.session_date, t.session_idx, t.turn_idx) for t in graph.turns}

    # Global chronological ordering across all sessions. The graph's
    # intra-session temporal edges plus the session_date+session_idx ordering
    # together induce this linear ordering; we materialize it once so a seed
    # can reach its time-adjacent neighbors that live in a different session.
    global_ordered = sorted(by_id.keys(), key=lambda tid: sort_key[tid])
    global_pos = {tid: i for i, tid in enumerate(global_ordered)}

    # Stable ranking: -score, then chronological. Zero-score turns end up last
    # and won't be picked until everything positive is exhausted.
    ranked = sorted(
        (t.turn_id for t in graph.turns),
        key=lambda tid: (-scores.get(tid, 0.0), sort_key[tid]),
    )

    selected: list[str] = []
    selected_set: set[str] = set()
    running = 0
    truncated = False
    n_seeds = 0
    n_expanded = 0

    def _add(tid: str, *, is_seed: bool) -> bool:
        nonlocal running, n_seeds, n_expanded, truncated
        if tid in selected_set:
            return True
        n = _tok_count(by_id[tid].text) + 2  # +2 approximates the "\n\n" joiner
        if running + n > token_budget:
            truncated = True
            return False
        selected.append(tid)
        selected_set.add(tid)
        running += n
        if is_seed:
            n_seeds += 1
        else:
            n_expanded += 1
        return True

    def _expansion_targets(seed_tid: str) -> list[str]:
        """Temporal neighbors of `seed_tid` in deterministic order: the
        intra-session paired turn first, then global session-date neighbors
        out to `temporal_expansion_hops`, in chronological order.
        """
        t = by_id[seed_tid]
        out: list[str] = []
        seen: set[str] = set()

        # Intra-session pair (preserved structural use).
        for neigh in (
            f"{t.session_id}#{t.turn_idx - 1}" if t.turn_idx > 0 else None,
            f"{t.session_id}#{t.turn_idx + 1}",
        ):
            if neigh and neigh in by_id and neigh not in seen:
                out.append(neigh)
                seen.add(neigh)

        # Global session-date temporal neighbors within `hops`. Skips the
        # seed itself; intra-session neighbors that overlap are already in
        # `seen` so they aren't appended twice. Order is chronological
        # (closest before, farthest after) for stable assembly.
        if temporal_expansion_hops > 0:
            i = global_pos[seed_tid]
            lo = max(0, i - temporal_expansion_hops)
            hi = min(len(global_ordered), i + temporal_expansion_hops + 1)
            for j in range(lo, hi):
                neigh = global_ordered[j]
                if neigh == seed_tid or neigh in seen:
                    continue
                out.append(neigh)
                seen.add(neigh)
        return out

    # Interleaved fill: for each positive-score seed (in score order), add the
    # seed AND its temporal expansions before moving to the next seed. The
    # budget is consumed in (seed, neighbors) bundles so a relevant turn's
    # temporal context cannot be evicted by a lower-relevance but
    # higher-similarity later seed. Zero-score seeds are dropped.
    for tid in ranked:
        if scores.get(tid, 0.0) <= 0.0 and n_seeds > 0:
            # Exhausted positive-signal turns; remaining turns have no
            # lexical/semantic overlap with the query and should not appear
            # except via temporal expansion of a stronger seed (which has
            # already happened by the time we get here).
            break
        added_seed = _add(tid, is_seed=True)
        if not added_seed:
            # No room for this seed; expansions of an unselected seed are
            # not meaningful, so move on. The score-ordered loop continues
            # in case a smaller later seed still fits — matches prior
            # behavior on the "ranked tail under tight budget" case.
            continue
        for neigh in _expansion_targets(tid):
            _add(neigh, is_seed=False)

    # Emit in chronological order.
    selected.sort(key=lambda tid: sort_key[tid])
    text = "\n\n".join(by_id[tid].text for tid in selected)
    return AssemblyResult(
        text=text,
        truncated=truncated,
        selected_turn_ids=selected,
        n_seeds=n_seeds,
        n_expanded=n_expanded,
    )
