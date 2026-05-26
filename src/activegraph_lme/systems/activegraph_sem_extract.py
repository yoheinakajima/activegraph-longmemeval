"""Stage 1 semantic-memory ActiveGraph system (no supersession).

Pipeline:
    ingest()
      1. Build the base turn graph via the package-native build_graph()
         (Turn + Session objects, precedes + cooccurrence relations).
      2. Register an @llm_behavior that reacts to one synthetic
         ``session.extract_request`` event per session and writes Fact
         objects + ``mentions`` relations through the same package
         ``ag.Graph``. Drive the reaction loop with
         ``runtime.run_until_idle()`` (single-threaded, FIFO).
      3. Facts are UNRESOLVED: contradictions coexist as separate Fact
         objects. Supersession (Stage 2) is intentionally out of scope.

    retrieve()
      Lexical IDF-overlap over the SAME pool that assemble() now iterates
      (turns + facts) — no separate "facts section". Same token budget as
      the deterministic systems so accuracy comparisons aren't confounded
      by context size.

Extractor / reader model identity
    The extractor @llm_behavior requests the SAME alias the reader
    requests (``claude-sonnet-4-5``); the API-resolved dated snapshot is
    captured per-call from the ``llm.responded`` event payload and stored
    in meta as ``extractor_model_resolved`` so a reviewer can confirm it
    matches (or doesn't match) the reader's resolved snapshot pinned in
    manifest.json. Extractor and reader intentionally share this alias;
    representational self-preference is a disclosed, uncontrolled
    residual (acceptable because the judge — not the reader — scores, so
    there is no scoring circularity). The extractor is non-GPT by
    construction (the judge is GPT-4o).

Run-identity invariants
    The reaction loop fires behaviors in registration order (FIFO,
    single-threaded). Replay is not reproducible without knowing which
    behaviors were registered, in what order; meta records
    ``behaviors_registered`` accordingly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any

import activegraph as ag
from pydantic import BaseModel, Field

from ..activegraph.graph import IngestState, Turn, build_graph
from ..activegraph.retrieve import (
    AssemblyResult,
    assemble,
    _tokenize_query,
)
from ..activegraph.stoplist import STOPLIST
from ..data import LMEInstance
from .base import AssembledContext


log = logging.getLogger(__name__)


_EXTRACT_REQUEST_TYPE = "session.extract_request"
_EXTRACTOR_BEHAVIOR_NAME = "sem_extract_facts_from_session"


# ---- LLM output schema ------------------------------------------------------


class _ExtractedFact(BaseModel):
    """One atomic fact the model believes the session establishes."""

    text: str = Field(
        ...,
        description=(
            "A single self-contained statement about the user or their world "
            "(<=160 chars). Do not include speculation, questions, or hedges."
        ),
    )
    mentioned_turn_idxs: list[int] = Field(
        default_factory=list,
        description=(
            "0-based indices of the turns in THIS session that establish the "
            "fact. Use [] only when the fact is implied by the session as a "
            "whole rather than any single turn."
        ),
    )


class _ExtractedFactList(BaseModel):
    facts: list[_ExtractedFact] = Field(default_factory=list)


# ---- extraction behavior (module-level so registration order is stable) ----


_EXTRACT_PROMPT_TEMPLATE = """\
{system}

Session under review:
{event}

Task:
{instruction}

Rules:
- Only extract facts that the session text directly supports.
- One claim per fact. Split compound claims.
- Extract ONLY facts about the user's world: their preferences,
  possessions, plans, identity, relationships, history, habits, opinions.
  Exclude facts that merely record the user asking the assistant a
  question, requesting help, or describe the topic of an
  assistant-provided answer. If the fact is only true because the user
  asked the assistant about something, exclude it. Include a fact only
  if it would still be true even if this conversation never happened.
  In particular, do NOT emit facts of the form "The user asked ...",
  "The user requested ...", "The user wanted to learn / know ...",
  "The user is learning how to ...", or "The user is interested in
  <topic the assistant explained>".
- Prefer durable facts about the user (preferences, identity, history,
  ongoing situations) over single-occasion small talk.
- Use neutral third-person phrasing ("The user ...").
- Do not emit two facts that express the same claim at different
  granularities; keep the most specific single version.
- If the session contains no extractable facts, return {{"facts": []}}.
"""


@ag.llm_behavior(
    name=_EXTRACTOR_BEHAVIOR_NAME,
    on=[_EXTRACT_REQUEST_TYPE],
    output_schema=_ExtractedFactList,
    # model=None -> runtime resolves to the configured provider's
    # default_model at registration time. AnthropicProvider's default is
    # "claude-sonnet-4-5" (the SAME alias the reader uses), per the
    # package's resolution rules. We do NOT hardcode a dated snapshot;
    # the actually-served snapshot is pulled from the llm.responded event
    # payload after run_until_idle().
    temperature=0.0,
    max_tokens=2048,
    timeout_seconds=60.0,
    description=(
        "Extract atomic facts from a single haystack session. Emits one "
        "Fact object per claim with a stable content-hash id and "
        "`mentions` relations to the supporting turns."
    ),
    prompt_template=_EXTRACT_PROMPT_TEMPLATE,
)
def _sem_extract_handler(
    event: ag.Event,
    bgraph: Any,
    ctx: Any,
    parsed: _ExtractedFactList,
) -> None:
    """Write Fact + mentions edges into the package graph.

    Determinism for replay: fact ids are content-hashed
    (``fact:<sha256(session_id|text)[:16]>``) so the same session text
    re-extracted into the same Fact text yields the same fact_id.
    """
    payload = event.payload or {}
    session_id = str(payload.get("session_id", ""))
    session_date = str(payload.get("session_date", ""))
    session_idx = int(payload.get("session_idx", 0))
    turn_object_ids: list[str] = list(payload.get("turn_object_ids", []))
    n_turns = len(turn_object_ids)

    for fact in parsed.facts:
        text = (fact.text or "").strip()
        if not text:
            continue
        h = hashlib.sha256(f"{session_id}|{text}".encode("utf-8")).hexdigest()[:16]
        fact_id = f"fact:{h}"

        fact_obj = bgraph.add_object(
            "Fact",
            {
                "fact_id": fact_id,
                "text": text,
                "session_id": session_id,
                "session_date": session_date,
                "session_idx": session_idx,
                "source": "llm_extract",
            },
        )

        for idx in fact.mentioned_turn_idxs:
            if not isinstance(idx, int) or idx < 0 or idx >= n_turns:
                continue
            bgraph.add_relation(fact_obj.id, turn_object_ids[idx], "mentions")


# ---- system state -----------------------------------------------------------


@dataclass
class _SemState:
    state: IngestState
    meta: dict[str, Any] = field(default_factory=dict)


# ---- system -----------------------------------------------------------------


class ActiveGraphSemExtractSystem:
    """Stage 1 semantic-memory ActiveGraph (no supersession).

    Public name: ``activegraph-sem-extract``.
    """

    name = "activegraph-sem-extract"

    def __init__(
        self,
        *,
        token_budget: int,
        min_token_length: int,
        min_session_cooccurrence: int,
        max_doc_freq_fraction: float,
        extractor_model: str = "claude-sonnet-4-5",
    ) -> None:
        self.token_budget = token_budget
        self.min_token_length = min_token_length
        self.min_session_cooccurrence = min_session_cooccurrence
        self.max_doc_freq_fraction = max_doc_freq_fraction
        # The alias requested at extraction call time. The runtime sends
        # this to AnthropicProvider; the dated snapshot the API serves
        # is what we record into meta.extractor_model_resolved.
        self.extractor_model = extractor_model

    # ---- ingest -------------------------------------------------------------

    def ingest(self, instance: LMEInstance) -> _SemState:
        ingest_state = build_graph(
            instance.haystack_session_ids,
            instance.haystack_dates,
            instance.haystack_sessions,
            min_token_length=self.min_token_length,
            min_session_cooccurrence=self.min_session_cooccurrence,
            max_doc_freq_fraction=self.max_doc_freq_fraction,
        )

        # Behavior registration order is a run-identity invariant — the
        # package's reaction loop is FIFO single-threaded and behaviors
        # fire in registration order. Record it so a reviewer can verify
        # replay would re-emit events identically.
        behaviors = [_sem_extract_handler]
        behaviors_registered = [
            {"name": b.name, "on": list(b.on), "model": b.model}
            for b in behaviors
        ]

        llm_provider = _build_llm_provider(self.extractor_model)

        runtime = ag.Runtime(
            ingest_state.graph,
            behaviors=behaviors,
            llm_provider=llm_provider,
            seed=0,
        )
        # Replace the IngestState's runtime stub with the live one so
        # downstream replay/debug walks the same instance.
        ingest_state.runtime = runtime

        # Emit one extract-request event per session, in input order.
        # The runtime's FIFO scheduler will then fan them through the
        # extraction behavior.
        sessions_by_id = self._group_sessions(ingest_state, instance)
        for sid, sdate, s_idx in zip(
            instance.haystack_session_ids,
            instance.haystack_dates,
            range(len(instance.haystack_session_ids)),
        ):
            turn_views = sessions_by_id[sid]
            session_text = "\n".join(
                f"[turn {tv.turn_idx}] {tv.role}: {tv.content}" for tv in turn_views
            )
            evt = ag.Event(
                id=ingest_state.graph.ids.event(),
                type=_EXTRACT_REQUEST_TYPE,
                payload={
                    "session_id": sid,
                    "session_date": sdate,
                    "session_idx": s_idx,
                    "session_text": session_text,
                    "turn_object_ids": [tv.object_id for tv in turn_views],
                    "n_turns": len(turn_views),
                },
                actor="sem_extract_ingest",
            )
            ingest_state.graph.emit(evt)

        # Drive the package's reaction loop. If extraction can't reach
        # the LLM (e.g. ANTHROPIC_API_KEY unset in this environment) the
        # behavior emits behavior.failed events and run_until_idle still
        # returns; we surface that to the caller via meta.n_facts == 0.
        try:
            runtime.run_until_idle()
        except Exception as exc:  # noqa: BLE001 — surface so harness can stop
            log.error("Extractor runtime.run_until_idle() raised: %r", exc)
            raise

        # Walk the post-run event log to pull the resolved extractor
        # snapshot (the dated model the API actually served).
        extractor_resolved = _resolve_extractor_snapshot(ingest_state.graph)
        n_facts = sum(1 for _ in ingest_state.graph.objects(type="Fact"))
        n_fact_events = sum(
            1 for e in ingest_state.graph.events if e.type == "object.created"
            and e.payload.get("object", {}).get("type") == "Fact"
        )
        n_mentions = len(ingest_state.graph.relations(type="mentions"))
        n_behavior_failed = sum(
            1 for e in ingest_state.graph.events if e.type == "behavior.failed"
        )

        meta = {
            **ingest_state.stats(),
            "n_facts": n_facts,
            "n_fact_events": n_fact_events,
            "n_mentions_edges": n_mentions,
            "n_behavior_failed": n_behavior_failed,
            "extractor_model_requested": self.extractor_model,
            "extractor_model_resolved": extractor_resolved,
            "behaviors_registered": behaviors_registered,
        }
        return _SemState(state=ingest_state, meta=meta)

    # ---- retrieve -----------------------------------------------------------

    def retrieve(
        self, state: _SemState, question: str, question_date: str
    ) -> AssembledContext:
        scores = _score_units_lexical(
            state.state, question, min_token_length=self.min_token_length
        )
        res: AssemblyResult = assemble(
            state.state, scores, token_budget=self.token_budget
        )
        # Partition selected ids by kind so the manifest meta surfaces both.
        selected_turn_ids = [uid for uid in res.selected_unit_ids if "#" in uid and not uid.startswith("fact:")]
        selected_fact_ids = [uid for uid in res.selected_unit_ids if uid.startswith("fact:")]
        meta = {
            **state.meta,
            "n_selected_turns": len(selected_turn_ids),
            "n_selected_facts": len(selected_fact_ids),
            "n_seeds": res.n_seeds,
            "n_temporal_expansions": res.n_expanded,
            "retrieval_signal": "lexical",
            "token_budget": self.token_budget,
        }
        return AssembledContext(text=res.text, truncated=res.truncated, meta=meta)

    # ---- helpers ------------------------------------------------------------

    def _group_sessions(
        self, state: IngestState, instance: LMEInstance
    ) -> dict[str, list[Turn]]:
        groups: dict[str, list[Turn]] = {sid: [] for sid in instance.haystack_session_ids}
        for t in state.turns:
            groups.setdefault(t.session_id, []).append(t)
        for sid in groups:
            groups[sid].sort(key=lambda tv: tv.turn_idx)
        return groups


# ---- scoring ----------------------------------------------------------------


def _score_units_lexical(
    state: IngestState, question: str, min_token_length: int
) -> dict[str, float]:
    """IDF-overlap score for every unit in the pool (turns + facts).

    Uses the existing turn-derived ``state.vocab_df`` for IDF weights so
    facts and turns share one yardstick. Facts get tokenized on the fly
    from their stored ``text``; turns reuse the cached ``turn_tokens``.
    Tokens outside the pruned vocab contribute 0 to both, exactly like
    score_lexical() does for turns alone.
    """
    q_tokens = _tokenize_query(question, min_token_length)
    n_turns = max(1, len(state.turns))
    idf: dict[str, float] = {}
    for tok in q_tokens:
        df = state.vocab_df.get(tok)
        if df is None:
            continue
        idf[tok] = math.log((n_turns + 1) / (df + 1)) + 1.0

    scores: dict[str, float] = {}
    for t in state.turns:
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

    for obj in state.graph.objects(type="Fact"):
        data = obj.data or {}
        fact_id = str(data.get("fact_id") or obj.id)
        text = str(data.get("text", ""))
        if not text or not idf:
            scores[fact_id] = 0.0
            continue
        toks = _tokenize_query(text, min_token_length)
        s = 0.0
        for tok in toks:
            w = idf.get(tok)
            if w is not None:
                s += w
        scores[fact_id] = s

    return scores


# ---- provider + resolved-snapshot capture -----------------------------------


def _build_llm_provider(alias: str) -> Any:
    """Construct the AnthropicProvider the runtime uses to drive
    ``@llm_behavior`` calls.

    The alias is recorded on the behavior at registration time; the
    actually-served dated snapshot is captured per-call by
    :func:`_resolve_extractor_snapshot` after run_until_idle.
    """
    from activegraph.llm import AnthropicProvider

    if alias.lower().startswith("gpt") or "openai" in alias.lower():
        raise ValueError(
            f"Extractor must be non-GPT (the judge is GPT-4o); got alias={alias!r}."
        )
    return AnthropicProvider()


def _resolve_extractor_snapshot(graph: ag.Graph) -> str | None:
    """Pull the dated model snapshot the Anthropic API actually served
    from the most recent ``llm.responded`` event emitted by the
    extractor behavior.

    Returns ``None`` if no extractor LLM call was made (e.g. provider
    misconfigured, all behaviors failed). All ``llm.responded`` events
    emitted by the extractor must report the same resolved snapshot
    within a single run — we assert that here so a mid-run model
    rotation can't silently slip in.
    """
    snapshots: set[str] = set()
    for ev in graph.events:
        if ev.type != "llm.responded":
            continue
        # Filter to events caused by our extractor behavior. Both the
        # actor and the behavior_name in the payload identify it.
        if ev.actor != _EXTRACTOR_BEHAVIOR_NAME and ev.payload.get(
            "behavior_name"
        ) != _EXTRACTOR_BEHAVIOR_NAME:
            continue
        model = ev.payload.get("model")
        if model:
            snapshots.add(str(model))
    if not snapshots:
        return None
    if len(snapshots) > 1:
        raise RuntimeError(
            "Extractor served multiple resolved snapshots in one ingest "
            f"({sorted(snapshots)!r}); refusing to record ambiguous provenance."
        )
    return next(iter(snapshots))
