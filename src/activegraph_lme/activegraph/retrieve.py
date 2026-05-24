"""Retrieval signals + budgeted assembly.

Two pinned signals share the same selection/assembly path. The signal
returns a per-Turn score; the assembler:

  1. ranks turns by (score desc, sort_key asc — chronological tie-break),
  2. greedily adds turns under ``token_budget`` (computed via project tokenizer),
  3. for each selected turn, adds its 1-hop temporal neighbor in the same
     session (chronologically adjacent), again under budget,
  4. emits selected turns in chronological order, joined by ``"\\n\\n"``.

Step 3 is the structural "graph use" that distinguishes this from
naive top-K — selecting an assistant reply pulls in the preceding user
turn (and vice versa) so the reader sees the local pair, not a bare
single utterance. It is deterministic and free of tuned knobs.
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
                _batch = [truncate_for_embedding(t) for t in new_texts[i : i + B]]
                resp = cli.embeddings.create(model=self.model, input=_batch)
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
) -> AssemblyResult:
    if not graph.turns:
        return AssemblyResult(text="", truncated=False, selected_turn_ids=[], n_seeds=0, n_expanded=0)

    by_id = {t.turn_id: t for t in graph.turns}
    sort_key = {t.turn_id: (t.session_date, t.session_idx, t.turn_idx) for t in graph.turns}

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

    def _fits(tid: str) -> bool:
        nonlocal running
        n = _tok_count(by_id[tid].text) + 2  # +2 approximates the "\n\n" joiner
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
            # Stop once we've exhausted positive-signal turns; the remaining
            # turns have no lexical/semantic overlap with the query and should
            # only appear via temporal expansion (or not at all).
            break
        if not _fits(tid):
            truncated = True
            continue
        if _add(tid):
            n_seeds += 1

    # 2) 1-hop temporal expansion in deterministic order over current seeds.
    n_expanded = 0
    expansion_targets: list[str] = []
    seeds_snapshot = list(selected)
    for tid in seeds_snapshot:
        t = by_id[tid]
        # previous turn in same session
        prev_id = f"{t.session_id}#{t.turn_idx - 1}" if t.turn_idx > 0 else None
        # next turn in same session
        next_id = f"{t.session_id}#{t.turn_idx + 1}"
        for neigh in (prev_id, next_id):
            if neigh is None:
                continue
            if neigh in selected_set:
                continue
            if neigh in by_id:
                expansion_targets.append(neigh)

    # Add expansions in chronological order so the result is deterministic
    # and time-ordered when the budget runs out mid-expansion.
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
