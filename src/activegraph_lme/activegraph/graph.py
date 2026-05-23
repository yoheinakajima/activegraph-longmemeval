"""Deterministic graph builder.

Every input turn becomes a Turn node. Edges are derived from corpus-relative,
fully deterministic rules:

  * ``temporal``      adjacency edges within a session (turn i — turn i+1)
  * ``cooccurrence``  cross-session edges between turns that share at least one
                      distinctive token, where a distinctive token must:
                        - have length >= ``min_token_length``,
                        - NOT be in the pinned stoplist,
                        - appear in ``>= min_session_cooccurrence`` turns across
                          DIFFERENT sessions (per-corpus statistic),
                        - appear in AT MOST ``max_doc_freq_fraction`` of all
                          turns (per-corpus statistic).

No per-bucket cap is applied — if edges blow up, the manifest reports it and
the user decides whether to tighten thresholds. There are no tuned knobs;
the three numbers above are corpus-statistics thresholds, frozen by config.

The builder emits an ordered event log. Re-building the same instance with
the same config under a FrozenClock + seed produces a byte-identical
``events_json()`` — the re-ingest-equality property test enforces this.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .stoplist import STOPLIST


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class Turn:
    turn_id: str          # "{session_id}#{turn_idx}"
    session_id: str
    session_date: str
    session_idx: int      # position of the session in the haystack
    turn_idx: int         # position of the turn within its session
    role: str
    content: str
    text: str             # rendered "[Session sid (date)] role: content"


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    kind: str             # "temporal" | "cooccurrence"
    weight: float


@dataclass
class Graph:
    turns: list[Turn] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    # Doc-frequency map AFTER pruning: token -> count of turns containing it.
    vocab_df: dict[str, int] = field(default_factory=dict)
    # Per-turn distinctive tokens, lowercased, deduped, sorted.
    turn_tokens: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # Ordered event log capturing every observable build action.
    events: list[tuple] = field(default_factory=list)

    def events_json(self) -> str:
        """Stable JSON serialization for the re-ingest-equality test."""
        return json.dumps(self.events, ensure_ascii=False, sort_keys=False)

    def stats(self) -> dict[str, Any]:
        n_temporal = sum(1 for e in self.edges if e.kind == "temporal")
        n_cooc = sum(1 for e in self.edges if e.kind == "cooccurrence")
        return {
            "n_turns": len(self.turns),
            "n_vocab": len(self.vocab_df),
            "n_edges_temporal": n_temporal,
            "n_edges_cooccurrence": n_cooc,
        }


def _tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def _render_turn(sid: str, date: str, role: str, content: str) -> str:
    return f"[Session {sid} ({date})] {role}: {content}"


def build_graph(
    haystack_session_ids: list[str],
    haystack_dates: list[str],
    haystack_sessions: list[list[dict[str, Any]]],
    *,
    min_token_length: int,
    min_session_cooccurrence: int,
    max_doc_freq_fraction: float,
) -> Graph:
    """Build a deterministic Graph from a LongMemEval haystack.

    Order is fixed: sessions in their input order, turns in their input order.
    """
    g = Graph()

    # 1) Materialize Turn nodes in input order.
    for s_idx, (sid, date, turns) in enumerate(
        zip(haystack_session_ids, haystack_dates, haystack_sessions)
    ):
        for t_idx, turn in enumerate(turns):
            role = str(turn.get("role", "?"))
            content = str(turn.get("content", ""))
            tid = f"{sid}#{t_idx}"
            g.turns.append(
                Turn(
                    turn_id=tid,
                    session_id=sid,
                    session_date=date,
                    session_idx=s_idx,
                    turn_idx=t_idx,
                    role=role,
                    content=content,
                    text=_render_turn(sid, date, role, content),
                )
            )
            g.events.append(("add_turn", tid, s_idx, t_idx, role))

    # 2) Tokenize each turn under the length+stoplist filter.
    raw_tokens: dict[str, list[str]] = {}
    df_all: dict[str, int] = {}
    df_sessions: dict[str, set[str]] = {}  # token -> set(session_id)
    for t in g.turns:
        toks: list[str] = []
        seen_in_turn: set[str] = set()
        for tok in _tokenize(t.content):
            if len(tok) < min_token_length:
                continue
            if tok in STOPLIST:
                continue
            if tok in seen_in_turn:
                continue
            seen_in_turn.add(tok)
            toks.append(tok)
        raw_tokens[t.turn_id] = toks
        for tok in toks:
            df_all[tok] = df_all.get(tok, 0) + 1
            df_sessions.setdefault(tok, set()).add(t.session_id)

    # 3) Corpus-relative pruning.
    n_turns = max(1, len(g.turns))
    max_df = int(n_turns * max_doc_freq_fraction)  # floor: integer threshold
    kept: set[str] = set()
    for tok, df in df_all.items():
        if df > max_df:
            continue
        if len(df_sessions.get(tok, ())) < min_session_cooccurrence:
            continue
        kept.add(tok)

    # 4) Record per-turn kept tokens and vocab DF, sorted for determinism.
    for t in g.turns:
        keep = tuple(sorted(tok for tok in raw_tokens[t.turn_id] if tok in kept))
        g.turn_tokens[t.turn_id] = keep
    g.vocab_df = {tok: df_all[tok] for tok in sorted(kept)}

    # 5) Temporal edges: turn_idx adjacency within each session.
    for s_idx, (sid, _, turns) in enumerate(
        zip(haystack_session_ids, haystack_dates, haystack_sessions)
    ):
        for t_idx in range(1, len(turns)):
            a = f"{sid}#{t_idx - 1}"
            b = f"{sid}#{t_idx}"
            g.edges.append(Edge(src=a, dst=b, kind="temporal", weight=1.0))
            g.events.append(("add_edge", "temporal", a, b))

    # 6) Co-occurrence edges across DIFFERENT sessions on the pruned vocab.
    #    Build token -> [turn_ids] in insertion (Turn) order.
    tok_to_turns: dict[str, list[str]] = {}
    for t in g.turns:
        for tok in g.turn_tokens[t.turn_id]:
            tok_to_turns.setdefault(tok, []).append(t.turn_id)

    # Map turn_id -> Turn for session_id lookup.
    by_id = {t.turn_id: t for t in g.turns}

    # Aggregate weights for unique (a, b) pairs across all shared tokens.
    pair_weight: dict[tuple[str, str], float] = {}
    for tok in sorted(tok_to_turns):
        tids = tok_to_turns[tok]
        # IDF-style edge weight contribution per shared token: 1 / df.
        df = max(1, g.vocab_df[tok])
        w = 1.0 / df
        # Pairwise within the same token bucket; skip same-session pairs.
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                a, b = tids[i], tids[j]
                if by_id[a].session_id == by_id[b].session_id:
                    continue
                key = (a, b) if a < b else (b, a)
                pair_weight[key] = pair_weight.get(key, 0.0) + w

    # Emit edges in deterministic order.
    for (a, b) in sorted(pair_weight.keys()):
        w = pair_weight[(a, b)]
        # Round to a stable precision so event-log equality survives
        # floating-point reassociation across re-runs.
        wr = round(w, 9)
        g.edges.append(Edge(src=a, dst=b, kind="cooccurrence", weight=wr))
        g.events.append(("add_edge", "cooccurrence", a, b, wr))

    return g
