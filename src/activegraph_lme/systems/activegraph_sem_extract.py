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
from activegraph.runtime.behavior_graph import BehaviorGraph
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


def _write_facts_to_graph(
    bgraph: Any,
    parsed: _ExtractedFactList,
    *,
    session_id: str,
    session_date: str,
    session_idx: int,
    turn_object_ids: list[str],
) -> None:
    """Write Fact + mentions edges for one session's extraction result.

    Shared by both the live @llm_behavior handler (on a cache miss,
    after the LLM call) and the in-memory session-extraction cache
    (on a cache hit, skipping the LLM call). Identical output by
    construction — both paths feed the same `parsed` shape into the
    same writes, so a cached hit produces byte-identical Fact ids and
    mentions edges to a live miss for the same session text.

    Determinism for replay: fact ids are content-hashed
    (``fact:<sha256(session_id|text)[:16]>``) so the same session text
    re-extracted into the same Fact text yields the same fact_id —
    regardless of whether the parsed value came from the live API or
    the cache.
    """
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
    """Live-API path: invoked by the runtime on a cache miss after the
    LLM has produced a parsed _ExtractedFactList. All Fact-writing is
    delegated to :func:`_write_facts_to_graph` so the cache-hit path
    (which skips the LLM but still must produce identical writes) calls
    exactly the same code.
    """
    payload = event.payload or {}
    _write_facts_to_graph(
        bgraph,
        parsed,
        session_id=str(payload.get("session_id", "")),
        session_date=str(payload.get("session_date", "")),
        session_idx=int(payload.get("session_idx", 0)),
        turn_object_ids=list(payload.get("turn_object_ids", [])),
    )


def _compute_extractor_signature(
    prompt_template: str, extractor_model_alias: str
) -> str:
    """Hash of the inputs that determine the LLM's parsed output.

    Used as the third element of the per-session cache key. If either
    the prompt template or the extractor model alias changes
    mid-process (shouldn't happen in one run, but defensively),
    cache lookups under the old signature will simply miss — stale
    entries can't be served.
    """
    h = hashlib.sha256()
    h.update(prompt_template.encode("utf-8"))
    h.update(b"\x00")
    h.update(extractor_model_alias.encode("utf-8"))
    return h.hexdigest()[:16]


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
        extraction_cache_enabled: bool | None = None,
    ) -> None:
        self.token_budget = token_budget
        self.min_token_length = min_token_length
        self.min_session_cooccurrence = min_session_cooccurrence
        self.max_doc_freq_fraction = max_doc_freq_fraction
        # The alias requested at extraction call time. The runtime sends
        # this to AnthropicProvider; the dated snapshot the API serves
        # is what we record into meta.extractor_model_resolved.
        self.extractor_model = extractor_model

        # In-memory per-run session-extraction cache. Lives on this
        # system instance — NOT module-global, NOT on disk. `cli run`
        # constructs one ActiveGraphSemExtractSystem and walks all
        # questions through it, so the cache scopes to a single CLI
        # invocation; a second `cli run` gets a fresh instance with an
        # empty cache (preserves determinism and reflects the current
        # prompt/model at process start).
        #
        # Key: (session_id, sha256(session_text)[:16], extractor_signature)
        # where extractor_signature hashes the prompt template + model alias,
        # so a prompt/model change mid-process can't serve a stale entry.
        # Value: the parsed _ExtractedFactList from the most recent live
        # extraction of that session under the current signature.
        if extraction_cache_enabled is None:
            extraction_cache_enabled = (
                os.environ.get("ACTIVEGRAPH_SEM_EXTRACT_CACHE", "1") != "0"
            )
        self._extraction_cache_enabled: bool = extraction_cache_enabled
        self._extraction_cache: dict[
            tuple[str, str, str], _ExtractedFactList
        ] = {}
        self._extractor_signature: str = _compute_extractor_signature(
            _EXTRACT_PROMPT_TEMPLATE, self.extractor_model
        )
        # Cumulative counters across all ingests on this instance
        # (per-ingest counts also go into each ingest's meta).
        self._cum_sessions_total: int = 0
        self._cum_sessions_extracted: int = 0
        self._cum_cache_hits: int = 0

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

        # Partition sessions into cache hits / misses BEFORE driving
        # the reaction loop. Misses emit a session.extract_request the
        # behavior fires on (live LLM call). Hits skip the emit (and so
        # skip the LLM call entirely) — we write their Facts directly
        # after run_until_idle, using the same _write_facts_to_graph
        # helper the live handler uses, so the per-session graph
        # mutations are identical to a live miss for the same session.
        sessions_by_id = self._group_sessions(ingest_state, instance)
        cache_hit_writes: list[tuple[dict, _ExtractedFactList]] = []
        miss_event_id_to_key: dict[str, tuple[str, str, str]] = {}
        n_sessions_total = 0
        n_cache_hits = 0
        n_sessions_extracted = 0

        for sid, sdate, s_idx in zip(
            instance.haystack_session_ids,
            instance.haystack_dates,
            range(len(instance.haystack_session_ids)),
        ):
            turn_views = sessions_by_id[sid]
            session_text = "\n".join(
                f"[turn {tv.turn_idx}] {tv.role}: {tv.content}" for tv in turn_views
            )
            n_sessions_total += 1

            text_hash = hashlib.sha256(session_text.encode("utf-8")).hexdigest()[:16]
            cache_key = (sid, text_hash, self._extractor_signature)
            cached: _ExtractedFactList | None = None
            if self._extraction_cache_enabled:
                cached = self._extraction_cache.get(cache_key)

            payload = {
                "session_id": sid,
                "session_date": sdate,
                "session_idx": s_idx,
                "session_text": session_text,
                "turn_object_ids": [tv.object_id for tv in turn_views],
                "n_turns": len(turn_views),
            }

            if cached is not None:
                # CACHE HIT: skip the extract_request emit so the
                # behavior never fires for this session. Defer the
                # Fact-writing until after run_until_idle so the
                # event log shows misses first (preserving the FIFO
                # order of live extractions), then hits.
                n_cache_hits += 1
                cache_hit_writes.append((payload, cached))
            else:
                # CACHE MISS: emit the event the @llm_behavior listens
                # to. The runtime will assemble the prompt, call the
                # provider, parse the structured output, and invoke
                # _sem_extract_handler with the parsed result.
                n_sessions_extracted += 1
                evt = ag.Event(
                    id=ingest_state.graph.ids.event(),
                    type=_EXTRACT_REQUEST_TYPE,
                    payload=payload,
                    actor="sem_extract_ingest",
                )
                miss_event_id_to_key[evt.id] = cache_key
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

        # Harvest miss results from the event log into the cache, then
        # write Facts for hits (using the same helper as the live
        # handler — identical graph mutations by construction).
        if self._extraction_cache_enabled and miss_event_id_to_key:
            self._harvest_misses_into_cache(
                ingest_state.graph, miss_event_id_to_key
            )

        # CACHE-HIT WRITES: still emit real object.created /
        # relation.created events through a BehaviorGraph tagged with
        # the extractor's actor name. The LLM call is skipped — the
        # graph mutations are not. This preserves the package's
        # replay/inspectability contract: every Fact in the graph has
        # a real object.created event sitting in graph.events.
        for payload, cached in cache_hit_writes:
            hit_bgraph = BehaviorGraph(
                ingest_state.graph,
                actor=_EXTRACTOR_BEHAVIOR_NAME,
                caused_by=None,
                frame_id=None,
            )
            _write_facts_to_graph(
                hit_bgraph,
                cached,
                session_id=str(payload["session_id"]),
                session_date=str(payload["session_date"]),
                session_idx=int(payload["session_idx"]),
                turn_object_ids=list(payload["turn_object_ids"]),
            )

        # Update cumulative counters on the system instance.
        self._cum_sessions_total += n_sessions_total
        self._cum_sessions_extracted += n_sessions_extracted
        self._cum_cache_hits += n_cache_hits

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
            # Session-extraction cache visibility. Per-ingest counts plus
            # cumulative-on-this-system-instance so a reviewer can
            # confirm `n_sessions_extracted + n_cache_hits == n_sessions_total`
            # and see the savings as the cache warms across questions.
            "extraction_cache_enabled": self._extraction_cache_enabled,
            "extractor_signature": self._extractor_signature,
            "n_sessions_total": n_sessions_total,
            "n_sessions_extracted": n_sessions_extracted,
            "n_cache_hits": n_cache_hits,
            "cum_sessions_total": self._cum_sessions_total,
            "cum_sessions_extracted": self._cum_sessions_extracted,
            "cum_cache_hits": self._cum_cache_hits,
        }
        return _SemState(state=ingest_state, meta=meta)

    # ---- cache harvest ------------------------------------------------------

    def _harvest_misses_into_cache(
        self,
        graph: ag.Graph,
        miss_event_id_to_key: dict[str, tuple[str, str, str]],
    ) -> None:
        """Walk the post-run event log and record each cache miss's
        parsed LLM output into ``self._extraction_cache`` so the next
        question's matching session hits the cache instead of the API.

        Event chain (per the runtime's @llm_behavior invocation path):
          extract_request  ← caused_by ←  llm.requested  ← caused_by ←  llm.responded
        We look up llm.responded events for our behavior, walk back via
        ``caused_by`` two hops, and pull the parsed payload.
        """
        events_by_id: dict[str, Any] = {e.id: e for e in graph.events}
        for ev in graph.events:
            if ev.type != "llm.responded":
                continue
            if ev.payload.get("behavior") != _EXTRACTOR_BEHAVIOR_NAME:
                continue
            req = events_by_id.get(ev.caused_by) if ev.caused_by else None
            if req is None or req.type != "llm.requested":
                continue
            extract_req = events_by_id.get(req.caused_by) if req.caused_by else None
            if extract_req is None or extract_req.type != _EXTRACT_REQUEST_TYPE:
                continue
            cache_key = miss_event_id_to_key.get(extract_req.id)
            if cache_key is None:
                continue
            parsed_payload = ev.payload.get("parsed")
            if parsed_payload is None:
                continue
            try:
                parsed = _ExtractedFactList.model_validate(parsed_payload)
            except Exception:  # noqa: BLE001 — defensive; bad parse means no caching
                continue
            self._extraction_cache[cache_key] = parsed

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
