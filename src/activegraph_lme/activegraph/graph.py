"""Package-native deterministic ingest (Mode A).

Builds the ActiveGraph projection on top of the real ``activegraph`` pip
package: every Turn / Session is an ``activegraph.Object``, every adjacency
or co-occurrence is an ``activegraph.Relation``, and the event log is
``activegraph.Graph.events`` (created by ``add_object`` / ``add_relation``
under a ``FrozenClock``).

Determinism contract:

  * Object ids ("Turn#1", "Turn#2", ...) and relation ids ("rel_001", ...)
    are scoped per-``Graph`` and assigned in insertion order, so they are
    stable for a given input.
  * ``run_id`` is derived from a SHA-256 of the haystack itself, so the
    same instance reproduces the same run_id (and therefore identical
    event provenance), while a different instance gets a different
    run_id (no shared-state bleed across questions).
  * ``FrozenClock`` pins every event timestamp.

Edge derivation matches the original spec exactly:

  * ``precedes``      adjacency edges within a session (turn i — turn i+1)
  * ``cooccurrence``  cross-session edges between turns sharing at least one
                      distinctive token, where a distinctive token must:
                        - have length >= ``min_token_length``,
                        - NOT be in the pinned stoplist,
                        - appear in ``>= min_session_cooccurrence`` turns
                          across DIFFERENT sessions,
                        - appear in AT MOST ``max_doc_freq_fraction`` of all
                          turns.

Re-ingesting the same instance with the same config produces a
byte-identical ``events_json()`` — the re-ingest equality property test
enforces this against ``graph.events`` (the package's own event log).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

import activegraph as ag

from .stoplist import STOPLIST


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class Turn:
    """Read-only view of a Turn object projected from the package graph.

    Carries both the human-readable identity (``turn_id`` =
    ``{session_id}#{turn_idx}``) and the package's object id
    (``object_id`` = ``"Turn#<n>"``) so retrieval can walk the package
    graph via ``relations()`` / ``neighborhood()``.
    """

    turn_id: str
    session_id: str
    session_date: str
    session_idx: int
    turn_idx: int
    role: str
    content: str
    text: str
    object_id: str

    @property
    def kind(self) -> str:
        # Compatibility with the old Edge-style introspection in stats().
        return "Turn"


@dataclass(frozen=True)
class _EdgeView:
    """Compatibility view over a package Relation, exposing the kind-name
    the original event log used (``temporal`` / ``cooccurrence``)."""

    src: str
    dst: str
    kind: str
    weight: float


class IngestState:
    """Wraps a package ``Graph`` + ``Runtime`` plus the scoring metadata
    derived deterministically from the corpus.

    The package graph is the source of truth for nodes and edges; the
    side caches (``turns``, ``by_object_id``, ``vocab_df``,
    ``turn_tokens``) are convenience projections built at ingest time
    that point INTO the package graph (every ``Turn`` view carries its
    package ``object_id``). Retrieval operates over the package graph
    via ``objects()`` / ``relations()`` / ``neighborhood()``.
    """

    def __init__(
        self,
        graph: ag.Graph,
        runtime: ag.Runtime,
        turns: list[Turn],
        session_object_ids: dict[str, str],
        vocab_df: dict[str, int],
        turn_tokens: dict[str, tuple[str, ...]],
    ) -> None:
        self.graph = graph
        self.runtime = runtime
        self.turns = turns
        self.session_object_ids = session_object_ids
        self.vocab_df = vocab_df
        self.turn_tokens = turn_tokens
        # Index by both id flavors for fast lookups during retrieval.
        self.by_turn_id: dict[str, Turn] = {t.turn_id: t for t in turns}
        self.by_object_id: dict[str, Turn] = {t.object_id: t for t in turns}

    # ---- compatibility surface used by stats / property test --------------

    @property
    def events(self) -> list:
        return self.graph.events

    @property
    def edges(self) -> list[_EdgeView]:
        out: list[_EdgeView] = []
        for r in self.graph.all_relations():
            kind = "temporal" if r.type == "precedes" else r.type
            weight = float(r.data.get("weight", 1.0)) if r.data else 1.0
            out.append(_EdgeView(src=r.source, dst=r.target, kind=kind, weight=weight))
        return out

    def events_json(self) -> str:
        """Stable JSON serialization for the re-ingest-equality test."""
        return json.dumps(
            [ev.to_dict() for ev in self.graph.events],
            ensure_ascii=False,
            sort_keys=False,
        )

    def stats(self) -> dict[str, Any]:
        n_temporal = len(self.graph.relations(type="precedes"))
        n_cooc = len(self.graph.relations(type="cooccurrence"))
        return {
            "n_turns": len(self.turns),
            "n_vocab": len(self.vocab_df),
            "n_edges_temporal": n_temporal,
            "n_edges_cooccurrence": n_cooc,
        }


# ---- helpers ----------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def _render_turn(sid: str, date: str, role: str, content: str) -> str:
    return f"[Session {sid} ({date})] {role}: {content}"


def _stable_run_id(
    haystack_session_ids: list[str],
    haystack_dates: list[str],
    haystack_sessions: list[list[dict[str, Any]]],
) -> str:
    """Hash the haystack so two ingests of the same instance share a run_id
    (byte-identical events) and different instances get distinct run_ids
    (no shared-state bleed)."""
    h = hashlib.sha256()
    payload = json.dumps(
        [haystack_session_ids, haystack_dates, haystack_sessions],
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    h.update(payload)
    return "run-" + h.hexdigest()[:16]


# ---- build ------------------------------------------------------------------


def build_graph(
    haystack_session_ids: list[str],
    haystack_dates: list[str],
    haystack_sessions: list[list[dict[str, Any]]],
    *,
    min_token_length: int,
    min_session_cooccurrence: int,
    max_doc_freq_fraction: float,
    clock_iso: str = "2026-05-15T10:32:01Z",
    seed: int = 0,
) -> IngestState:
    """Build a package-native ActiveGraph from a LongMemEval haystack.

    Order is fixed: sessions in input order, turns in input order. Every
    node and edge is emitted via the real ``activegraph`` API.
    """
    run_id = _stable_run_id(haystack_session_ids, haystack_dates, haystack_sessions)
    graph = ag.Graph(clock=ag.FrozenClock(clock_iso), run_id=run_id)
    runtime = ag.Runtime(graph, seed=seed)

    # 1) Materialize Session and Turn objects in input order.
    turns: list[Turn] = []
    session_object_ids: dict[str, str] = {}
    raw_tokens: dict[str, list[str]] = {}

    for s_idx, (sid, date, session_turns) in enumerate(
        zip(haystack_session_ids, haystack_dates, haystack_sessions)
    ):
        sess_obj = graph.add_object(
            "Session",
            {"session_id": sid, "session_date": date, "session_idx": s_idx},
        )
        session_object_ids[sid] = sess_obj.id

        for t_idx, turn in enumerate(session_turns):
            role = str(turn.get("role", "?"))
            content = str(turn.get("content", ""))
            tid = f"{sid}#{t_idx}"
            text = _render_turn(sid, date, role, content)
            obj = graph.add_object(
                "Turn",
                {
                    "turn_id": tid,
                    "session_id": sid,
                    "session_date": date,
                    "session_idx": s_idx,
                    "turn_idx": t_idx,
                    "role": role,
                    "content": content,
                    "text": text,
                },
            )
            turns.append(
                Turn(
                    turn_id=tid,
                    session_id=sid,
                    session_date=date,
                    session_idx=s_idx,
                    turn_idx=t_idx,
                    role=role,
                    content=content,
                    text=text,
                    object_id=obj.id,
                )
            )
            graph.add_relation(sess_obj.id, obj.id, "contains")

            # Tokenize once; pruning decides whether tokens go into the
            # Turn object's `tokens` payload below.
            toks: list[str] = []
            seen_in_turn: set[str] = set()
            for tok in _tokenize(content):
                if len(tok) < min_token_length:
                    continue
                if tok in STOPLIST:
                    continue
                if tok in seen_in_turn:
                    continue
                seen_in_turn.add(tok)
                toks.append(tok)
            raw_tokens[tid] = toks

    # 2) Corpus-relative DF stats and pruning (deterministic).
    df_all: dict[str, int] = {}
    df_sessions: dict[str, set[str]] = {}
    by_turn_id: dict[str, Turn] = {t.turn_id: t for t in turns}
    for t in turns:
        for tok in raw_tokens[t.turn_id]:
            df_all[tok] = df_all.get(tok, 0) + 1
            df_sessions.setdefault(tok, set()).add(t.session_id)

    n_turns = max(1, len(turns))
    max_df = int(n_turns * max_doc_freq_fraction)
    kept: set[str] = set()
    for tok, df in df_all.items():
        if df > max_df:
            continue
        if len(df_sessions.get(tok, ())) < min_session_cooccurrence:
            continue
        kept.add(tok)

    turn_tokens: dict[str, tuple[str, ...]] = {}
    for t in turns:
        keep = tuple(sorted(tok for tok in raw_tokens[t.turn_id] if tok in kept))
        turn_tokens[t.turn_id] = keep
    vocab_df: dict[str, int] = {tok: df_all[tok] for tok in sorted(kept)}

    # 3) Temporal edges: turn_idx adjacency within each session.
    for s_idx, (sid, _, session_turns) in enumerate(
        zip(haystack_session_ids, haystack_dates, haystack_sessions)
    ):
        for t_idx in range(1, len(session_turns)):
            a = by_turn_id[f"{sid}#{t_idx - 1}"]
            b = by_turn_id[f"{sid}#{t_idx}"]
            graph.add_relation(a.object_id, b.object_id, "precedes")

    # 4) Co-occurrence edges across DIFFERENT sessions on the pruned vocab.
    #    Token order is sorted for determinism so the aggregation order is
    #    stable; pair-aggregation guarantees one relation per unique (a, b).
    tok_to_turns: dict[str, list[Turn]] = {}
    for t in turns:
        for tok in turn_tokens[t.turn_id]:
            tok_to_turns.setdefault(tok, []).append(t)

    pair_weight: dict[tuple[str, str], float] = {}  # keyed by turn_id pair
    for tok in sorted(tok_to_turns):
        tids = tok_to_turns[tok]
        df = max(1, vocab_df[tok])
        w = 1.0 / df
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                a, b = tids[i], tids[j]
                if a.session_id == b.session_id:
                    continue
                if a.turn_id < b.turn_id:
                    key = (a.turn_id, b.turn_id)
                else:
                    key = (b.turn_id, a.turn_id)
                pair_weight[key] = pair_weight.get(key, 0.0) + w

    for (a_tid, b_tid) in sorted(pair_weight.keys()):
        wr = round(pair_weight[(a_tid, b_tid)], 9)
        a_obj = by_turn_id[a_tid].object_id
        b_obj = by_turn_id[b_tid].object_id
        graph.add_relation(a_obj, b_obj, "cooccurrence", data={"weight": wr})

    return IngestState(
        graph=graph,
        runtime=runtime,
        turns=turns,
        session_object_ids=session_object_ids,
        vocab_df=vocab_df,
        turn_tokens=turn_tokens,
    )
