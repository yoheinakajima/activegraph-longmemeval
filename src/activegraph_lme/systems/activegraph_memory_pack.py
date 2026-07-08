"""Adapter for the external ``activegraph-memory`` pack.

This benchmark cell is intentionally conservative for the pack's 0.1
surface: it keeps LongMemEval's reader context as conversation history only,
while exercising the pack's deterministic query planning, coverage reporting,
confidence metadata, and memory_gateway request shaping.

The retrieval substrate is the existing deterministic ActiveGraph lexical
graph. That means Phase 1 should be interpreted as an integration and
instrumentation baseline, not as a semantic-memory improvement claim. As the
pack gains claim extraction, supersession, and evidence-bundle runtime
behaviors, this adapter is the stable place to plug them into LongMemEval.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..data import LMEInstance
from .activegraph_det import ActiveGraphDetSystem
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
        _load_activegraph_memory_module("activegraph_memory.planner")
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    return True, ""


@dataclass
class _State:
    det_state: Any
    session_ids: list[str]
    question_id: str
    question_type: str


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
    ) -> None:
        self._det = ActiveGraphDetSystem(
            retrieval_signal="lexical",
            token_budget=token_budget,
            min_token_length=min_token_length,
            min_session_cooccurrence=min_session_cooccurrence,
            max_doc_freq_fraction=max_doc_freq_fraction,
        )

    def ingest(self, instance: LMEInstance) -> _State:
        return _State(
            det_state=self._det.ingest(instance),
            session_ids=list(instance.haystack_session_ids),
            question_id=instance.question_id,
            question_type=instance.question_type,
        )

    def retrieve(
        self, state: _State, question: str, question_date: str
    ) -> AssembledContext:
        constants = _load_activegraph_memory_module("activegraph_memory.constants")
        coverage_mod = _load_activegraph_memory_module("activegraph_memory.coverage")
        gateway_mod = _load_activegraph_memory_module("activegraph_memory.gateway")
        object_types = _load_activegraph_memory_module("activegraph_memory.object_types")
        planner = _load_activegraph_memory_module("activegraph_memory.planner")
        settings_mod = _load_activegraph_memory_module("activegraph_memory.settings")

        query_id = _query_id(state.question_id, question, question_date)
        inferred_type = planner.infer_query_type(question)
        memory_query = object_types.MemoryQuery(
            query=question,
            query_type=inferred_type,
            time_anchor=question_date or None,
            top_k=10,
            metadata={
                "benchmark": "longmemeval",
                "question_id": state.question_id,
                "question_type": state.question_type,
            },
        )
        memory_settings = settings_mod.ActiveGraphMemorySettings(
            default_top_k=10,
            enable_gateway_integration=True,
        )
        retrieval_plan = planner.plan_query(
            memory_query,
            query_id=query_id,
            settings=memory_settings,
        )

        ctx = self._det.retrieve(state.det_state, question, question_date)
        selected_unit_ids = list((ctx.meta or {}).get("selected_unit_ids") or [])
        searched_sessions = _session_ids_from_units(selected_unit_ids)
        not_searched = [sid for sid in state.session_ids if sid not in searched_sessions]
        coverage = coverage_mod.build_coverage_report(
            query_id=query_id,
            searched_scopes=searched_sessions,
            not_searched_scopes=not_searched,
            query_type=retrieval_plan.metadata.get("query_type", "unknown"),
            metadata={
                "scope_kind": "LongMemEval session ids",
                "selected_unit_ids": selected_unit_ids,
            },
        )
        confidence = _phase1_confidence(
            constants.CONFIDENCE_DIMENSIONS,
            coverage_confidence=coverage.coverage_confidence,
        )
        gateway_request = gateway_mod.build_memory_retrieval_request(
            memory_query,
            retrieval_plan,
            backend_url="longmemeval://deterministic-activegraph",
        )

        meta = {
            **(ctx.meta or {}),
            "system": self.name,
            "activegraph_memory_version": getattr(constants, "PACK_VERSION", "unknown"),
            "memory_query": memory_query.model_dump(),
            "retrieval_plan": retrieval_plan.model_dump(),
            "coverage_report": coverage.model_dump(),
            "confidence": confidence,
            "gateway_request": gateway_request,
            "phase1_contract": (
                "Conversation context is unchanged; activegraph-memory contributes "
                "deterministic planning, coverage, confidence, and gateway metadata."
            ),
        }
        return AssembledContext(text=ctx.text, truncated=ctx.truncated, meta=meta)


def _query_id(question_id: str, question: str, question_date: str) -> str:
    digest = hashlib.sha256(
        f"{question_id}\n{question_date}\n{question}".encode("utf-8")
    ).hexdigest()[:16]
    return f"lme:{question_id}:{digest}"


def _session_ids_from_units(unit_ids: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for uid in unit_ids:
        if "#" not in uid:
            continue
        sid = uid.rsplit("#", 1)[0]
        if sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


def _phase1_confidence(
    dimensions: tuple[str, ...],
    *,
    coverage_confidence: float,
) -> dict[str, float]:
    base = {dimension: 0.0 for dimension in dimensions}
    base.update(
        {
            "relevance": 0.5,
            "entity_match": 0.5,
            "authority": 0.5,
            "freshness": 0.5,
            "coverage": coverage_confidence,
            "consistency": 0.5,
            "extraction": 1.0,
            "reasoning": 0.5,
        }
    )
    return base
