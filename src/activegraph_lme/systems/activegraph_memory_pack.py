"""Adapter for the external ``activegraph-memory`` pack.

This cell now exercises the full pack runtime rather than only the Phase-1
planning metadata. The benchmark still gives the reader raw conversation
provenance, but retrieval is driven by a compiled memory index:

    source turns + extracted claims -> MemoryIndex -> evidence bundle

For LongMemEval we use the existing ``data/sem_extract_cache/seed-A-v2.jsonl``
claim cache when present. That cache contains role-aware user and assistant
facts from the earlier semantic-memory experiments. Missing cache entries
fall back to deterministic turn-derived claims so offline contract tests do
not need Anthropic.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..activegraph.graph import IngestState, Turn, build_graph
from ..activegraph.retrieve import EmbeddingClient
from ..data import LMEInstance
from ..tokens import count_tokens
from .base import AssembledContext


def _load_activegraph_memory_module(module_name: str):
    """Import activegraph-memory, falling back to a sibling checkout."""

    try:
        return importlib.import_module(module_name)
    except Exception as first_error:  # noqa: BLE001 - kept in the error below.
        repo_root = Path(__file__).resolve().parents[3]
        sibling = repo_root.parent / "activegraph-memory"
        if sibling.exists() and str(sibling) not in sys.path:
            sys.path.insert(0, str(sibling))
        try:
            return importlib.import_module(module_name)
        except Exception as second_error:  # noqa: BLE001
            raise RuntimeError(
                "activegraph-memory is required for system 'activegraph-memory-pack'. "
                "Install it with `pip install -e ../activegraph-memory`, or keep a "
                "sibling checkout named activegraph-memory. Original import errors: "
                f"{first_error!r}; {second_error!r}"
            ) from second_error


def activegraph_memory_available() -> tuple[bool, str]:
    """Return whether the external pack can be imported in this environment."""

    try:
        _load_activegraph_memory_module("activegraph_memory.retrieval")
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    return True, ""


@dataclass
class _State:
    graph_state: IngestState
    memory_index: Any
    question_id: str
    question_type: str
    meta: dict[str, Any]


class ActiveGraphMemoryPackSystem:
    """LongMemEval adapter for the external activegraph-memory package."""

    name = "activegraph-memory-pack"

    def __init__(
        self,
        *,
        token_budget: int,
        min_token_length: int,
        min_session_cooccurrence: int,
        max_doc_freq_fraction: float,
        embedding_model: str = "text-embedding-3-small",
        extraction_cache_dir: str | Path | None = None,
        extract_seed: str = "A-v2",
    ) -> None:
        self.token_budget = token_budget
        self.min_token_length = min_token_length
        self.min_session_cooccurrence = min_session_cooccurrence
        self.max_doc_freq_fraction = max_doc_freq_fraction
        self.embedding_model = embedding_model
        self.extract_seed = extract_seed
        self.extraction_cache_dir = Path(extraction_cache_dir or "data/sem_extract_cache")
        self._extract_cache = _load_extract_cache(
            self.extraction_cache_dir / f"seed-{extract_seed}.jsonl"
        )
        self._embedder: EmbeddingClient | None = None

    def _get_embedder(self) -> EmbeddingClient:
        if self._embedder is None:
            self._embedder = EmbeddingClient(model=self.embedding_model)
        return self._embedder

    def ingest(self, instance: LMEInstance) -> _State:
        compiler = _load_activegraph_memory_module("activegraph_memory.compiler")

        graph_state = build_graph(
            instance.haystack_session_ids,
            instance.haystack_dates,
            instance.haystack_sessions,
            min_token_length=self.min_token_length,
            min_session_cooccurrence=self.min_session_cooccurrence,
            max_doc_freq_fraction=self.max_doc_freq_fraction,
        )
        source_turns = _source_turns(graph_state.turns)
        claim_inputs, cache_stats = _claim_inputs_from_cache_or_turns(
            instance,
            graph_state.turns,
            self._extract_cache,
            compiler.ExtractedClaimInput,
        )
        memory_index = compiler.compile_memory_index(
            turns=source_turns,
            claims=claim_inputs,
            metadata={
                "benchmark": "longmemeval",
                "question_id": instance.question_id,
                "question_type": instance.question_type,
                "extract_seed": self.extract_seed,
            },
        )
        meta = {
            **graph_state.stats(),
            **cache_stats,
            "n_memory_claims": len(memory_index.claims),
            "n_memory_turns": len(memory_index.turns),
            "memory_runtime": "activegraph_memory.compiler_retrieval_v1",
            "embedding_model": self.embedding_model,
            "extract_seed": self.extract_seed,
            "extraction_cache_path": str(
                self.extraction_cache_dir / f"seed-{self.extract_seed}.jsonl"
            ),
        }
        return _State(
            graph_state=graph_state,
            memory_index=memory_index,
            question_id=instance.question_id,
            question_type=instance.question_type,
            meta=meta,
        )

    def retrieve(
        self, state: _State, question: str, question_date: str
    ) -> AssembledContext:
        gateway_mod = _load_activegraph_memory_module("activegraph_memory.gateway")
        object_types = _load_activegraph_memory_module("activegraph_memory.object_types")
        retrieval_mod = _load_activegraph_memory_module("activegraph_memory.retrieval")

        query_id = _query_id(state.question_id, question, question_date)
        memory_query = object_types.MemoryQuery(
            query=question,
            time_anchor=question_date or None,
            top_k=20,
            metadata={
                "benchmark": "longmemeval",
                "question_id": state.question_id,
                "question_type": state.question_type,
            },
        )
        claim_scores, turn_scores, score_meta = self._score_index(
            state.memory_index, question
        )
        result = retrieval_mod.retrieve_memory(
            state.memory_index,
            memory_query,
            query_id=query_id,
            question_date=question_date,
            token_budget=self.token_budget,
            claim_scores=claim_scores,
            turn_scores=turn_scores,
            token_counter=count_tokens,
        )
        gateway_request = gateway_mod.build_memory_retrieval_request(
            memory_query,
            result.retrieval_plan,
            backend_url="longmemeval://compiled-activegraph-memory",
        )

        meta = {
            **state.meta,
            **score_meta,
            "system": self.name,
            "query_type": result.metadata.get("query_type"),
            "retrieval_plan": result.retrieval_plan.model_dump(),
            "coverage_report": result.coverage_report.model_dump(),
            "evidence_bundle": result.evidence_bundle.model_dump(),
            "confidence": result.confidence.as_answer_confidence(),
            "epistemic_status": result.epistemic_status,
            "selected_turn_ids": list(result.selected_turn_ids),
            "selected_claim_ids": list(result.selected_claim_ids),
            "selected_unit_ids": list(result.metadata.get("selected_unit_ids", [])),
            "gateway_request": gateway_request,
            "memory_retrieval_metadata": result.metadata,
        }
        return AssembledContext(
            text=result.context_text,
            truncated=result.truncated,
            meta=meta,
        )

    def _score_index(self, memory_index: Any, question: str) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
        if not os.environ.get("OPENAI_API_KEY"):
            return {}, {}, {"retrieval_signal": "lexical-pack-fallback"}

        embedder = self._get_embedder()
        claims = list(memory_index.claims)
        turns = list(memory_index.turns)
        q_vec = embedder.embed([question])[0]
        claim_scores: dict[str, float] = {}
        turn_scores: dict[str, float] = {}

        if claims:
            claim_vecs = embedder.embed([record.text for record in claims])
            raw = claim_vecs @ q_vec
            claim_scores = {
                record.claim_id: float(raw[i]) for i, record in enumerate(claims)
            }
        if turns:
            turn_vecs = embedder.embed([turn.text for turn in turns])
            raw = turn_vecs @ q_vec
            turn_scores = {
                turn.turn_id: float(raw[i]) for i, turn in enumerate(turns)
            }
        return claim_scores, turn_scores, {
            "retrieval_signal": "compiled-memory+embedding",
            "n_claim_scores": len(claim_scores),
            "n_turn_scores": len(turn_scores),
        }


def _load_extract_cache(path: Path) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    out: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    if not path.exists():
        return out
    with open(path) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            sid = str(obj["session_id"])
            csum = str(obj["content_sha256"])
            role = str(obj.get("role", "user"))
            facts = list((obj.get("parsed") or {}).get("facts") or [])
            out[(sid, csum, role)] = facts
    return out


def _source_turns(turns: list[Turn]) -> list[Any]:
    compiler = _load_activegraph_memory_module("activegraph_memory.compiler")
    return [
        compiler.SourceTurn(
            turn_id=turn.turn_id,
            session_id=turn.session_id,
            session_date=turn.session_date,
            session_idx=turn.session_idx,
            turn_idx=turn.turn_idx,
            role=turn.role,
            content=turn.content,
            text=turn.text,
            metadata={"object_id": turn.object_id},
        )
        for turn in turns
    ]


def _claim_inputs_from_cache_or_turns(
    instance: LMEInstance,
    turns: list[Turn],
    cache: dict[tuple[str, str, str], list[dict[str, Any]]],
    claim_cls: Any,
) -> tuple[list[Any], dict[str, Any]]:
    turns_by_session: dict[str, list[Turn]] = {}
    for turn in turns:
        turns_by_session.setdefault(turn.session_id, []).append(turn)
    out: list[Any] = []
    n_cache_hits = 0
    n_cache_misses = 0
    n_fallback_claims = 0
    n_cached_facts = 0

    for s_idx, (sid, sdate) in enumerate(
        zip(instance.haystack_session_ids, instance.haystack_dates)
    ):
        session_turns = turns_by_session.get(sid, [])
        session_text = "\n".join(
            f"[turn {turn.turn_idx}] {turn.role}: {turn.content}"
            for turn in session_turns
        )
        csum = hashlib.sha256(session_text.encode("utf-8")).hexdigest()
        found_for_session = False
        for role in ("user", "assistant"):
            facts = cache.get((sid, csum, role))
            if facts is None:
                continue
            found_for_session = True
            n_cache_hits += 1
            n_cached_facts += len(facts)
            for fact in facts:
                out.append(
                    claim_cls(
                        text=str(fact.get("text", "")),
                        session_id=sid,
                        session_date=sdate,
                        session_idx=s_idx,
                        role=role,
                        mentioned_turn_idxs=tuple(
                            idx for idx in fact.get("mentioned_turn_idxs", []) if isinstance(idx, int)
                        ),
                        confidence=0.86,
                        source="longmemeval_sem_extract_cache",
                    )
                )
        if found_for_session:
            continue

        n_cache_misses += 1
        for turn in session_turns:
            prefix = "The assistant said" if turn.role == "assistant" else "The user said"
            out.append(
                claim_cls(
                    text=f"{prefix}: {turn.content}",
                    session_id=sid,
                    session_date=sdate,
                    session_idx=s_idx,
                    role=turn.role,
                    mentioned_turn_idxs=(turn.turn_idx,),
                    confidence=0.45,
                    source="deterministic_turn_fallback",
                )
            )
            n_fallback_claims += 1

    return out, {
        "n_extract_cache_entries_loaded": len(cache),
        "n_extract_cache_role_hits": n_cache_hits,
        "n_extract_cache_session_misses": n_cache_misses,
        "n_cached_facts_used": n_cached_facts,
        "n_fallback_claims": n_fallback_claims,
    }


def _query_id(question_id: str, question: str, question_date: str) -> str:
    digest = hashlib.sha256(
        f"{question_id}\n{question_date}\n{question}".encode("utf-8")
    ).hexdigest()[:16]
    return f"lme:{question_id}:{digest}"

