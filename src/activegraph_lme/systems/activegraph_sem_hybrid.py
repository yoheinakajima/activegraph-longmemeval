"""``activegraph-sem-hybrid`` — variant (a): facts as headers + anchored turns.

assemble() scores facts by embedding cosine, greedy-selects them under the
2,500-token budget, and for each selected fact follows its ``mentions``
edges to ALSO include the source turn(s). The reader sees each fact as a
labeled header sitting directly above the raw turn(s) that establish it —
the semantic index and the verbatim provenance, together.

Two design choices, made explicitly (see the budget + render comments
inline below):

  1. Budget split — unified pool, scored greedy. A fact's marginal cost
     against the budget is ``tokens(fact-header) + tokens(each NOT-YET-
     -included provenance turn)``. High-scored facts therefore consume
     more budget because they drag their turn(s) in with them — biased
     toward fewer-but-deeper context, the right direction for the
     fidelity argument. Shared turns are charged ONCE (the second fact to
     reach a turn already in context pays only for its own header), so the
     rendered context stays within the same 2,500-token envelope as
     det-embedding rather than silently under-filling it. This is the
     "all selected facts bring their turns" default. The cheaper
     alternative (only the top-K=10 facts expand their turns; the rest go
     headers-only) is gated behind ``HYBRID_TOPK_EXPAND`` and is the
     documented downgrade to apply only if a measured smoke run shows
     fewer than 5 facts selected per question (recorded as
     ``budget_mode`` / ``n_facts_selected`` in meta).

  2. Render format — for each selected entry, emit
     ``[fact: <text>]\\n<turn text>``. When several facts share a turn the
     turn is rendered ONCE with all its fact-headers stacked above it.
     Entries are emitted in chronological order by the anchor turn's
     ``(session_date, session_idx, turn_idx)``; a selected fact with no
     provenance turn renders as a standalone header sorted by its own
     ``sort_key``.
"""

from __future__ import annotations

import os
from collections import defaultdict

from ..tokens import count_tokens
from .base import AssembledContext
from ._sem_compiled import _FactView, _SemCompiledBase, project_facts


# Top-K facts that expand their provenance turns under the downgrade mode.
# Only consulted when ACTIVEGRAPH_HYBRID_TOPK_EXPAND=1 (the documented
# fallback for under-filled budgets). Default mode ignores this entirely.
_TOPK_EXPAND = 10


class ActiveGraphSemHybridSystem(_SemCompiledBase):
    """Variant (a): facts-as-headers anchored above their provenance turns."""

    name = "activegraph-sem-hybrid"

    def retrieve(
        self, state, question: str, question_date: str
    ) -> AssembledContext:
        facts = project_facts(state.state)
        scores = self._score_facts(question, facts)
        return self._assemble(state, facts, scores)

    # ---- pure assembly (deterministic given the injected scores) -----------

    def _assemble(self, state, facts: list[_FactView], scores) -> AssembledContext:
        turn_by_id = state.state.by_turn_id
        ranked = self._rank_facts(facts, scores)

        # Downgrade switch: only the top-K selected facts expand their turns
        # when ACTIVEGRAPH_HYBRID_TOPK_EXPAND=1; everything below goes
        # headers-only. Default ("all selected facts bring turns") is False.
        topk_only = os.environ.get("ACTIVEGRAPH_HYBRID_TOPK_EXPAND") == "1"
        budget_mode = "topk10-expand" if topk_only else "all-selected-bring-turns"

        selected_facts: list[_FactView] = []
        selected_turn_ids: set[str] = set()
        running = 0
        truncated = False
        n_facts_considered = 0

        for f in ranked:
            n_facts_considered += 1
            header = f"[fact: {f.text}]"
            # +1 ≈ the newline between header and its turn / next header.
            cost = count_tokens(header) + 1

            expand = (not topk_only) or (len(selected_facts) < _TOPK_EXPAND)
            new_turns: list[str] = []
            if expand:
                for tid in f.turn_ids:
                    if tid in selected_turn_ids or tid not in turn_by_id:
                        continue
                    new_turns.append(tid)
                    # +1 ≈ the newline after each anchored turn.
                    cost += count_tokens(turn_by_id[tid].text) + 1

            if running + cost > self.token_budget:
                # Budget forced dropping this fact; keep scanning so a
                # cheaper lower-ranked fact can still fill remaining budget
                # (mirrors retrieve.assemble()'s greedy continue).
                truncated = True
                continue

            running += cost
            selected_facts.append(f)
            selected_turn_ids.update(new_turns)

        text = self._render(selected_facts, selected_turn_ids, turn_by_id)

        # Chronological turn-id list for the sidecar / AIC scoring.
        rendered_turn_ids = sorted(
            selected_turn_ids, key=lambda tid: turn_by_id[tid].sort_key
        )
        n_turns_anchored = sum(len(f.turn_ids) for f in selected_facts)

        meta = {
            **state.meta,
            "retrieval_signal": "embedding",
            "assembly": "sem-hybrid",
            "budget_mode": budget_mode,
            "token_budget": self.token_budget,
            "n_facts_selected": len(selected_facts),
            "n_facts_considered": n_facts_considered,
            "n_turns_anchored": n_turns_anchored,
            "n_unique_turns_rendered": len(selected_turn_ids),
            # Consumed by aic_sidecar.py (turn ids drive AIC hit-rate).
            "selected_turn_ids": rendered_turn_ids,
            "selected_fact_ids": [f.fact_id for f in selected_facts],
        }
        return AssembledContext(text=text, truncated=truncated, meta=meta)

    @staticmethod
    def _render(
        selected_facts: list[_FactView],
        selected_turn_ids: set[str],
        turn_by_id,
    ) -> str:
        """Render anchored entries in chronological order.

        Each rendered turn carries all selected facts that mention it,
        stacked as ``[fact: ...]`` headers above the single (de-duplicated)
        turn text. Selected facts with no provenance turn render as
        standalone headers, ordered by their own ``sort_key`` so they
        interleave chronologically with the turn blocks.
        """
        facts_for_turn: dict[str, list[_FactView]] = defaultdict(list)
        factless: list[_FactView] = []
        for f in selected_facts:
            anchored = [tid for tid in f.turn_ids if tid in selected_turn_ids]
            if anchored:
                for tid in anchored:
                    facts_for_turn[tid].append(f)
            else:
                factless.append(f)

        # (sort_key, block) entries; one block per rendered turn + one per
        # factless fact. sort_key totally orders turns and facts together.
        entries: list[tuple[tuple, str]] = []
        for tid in selected_turn_ids:
            tv = turn_by_id[tid]
            fs = sorted(facts_for_turn[tid], key=lambda f: f.sort_key)
            headers = "\n".join(f"[fact: {f.text}]" for f in fs)
            block = f"{headers}\n{tv.text}" if headers else tv.text
            entries.append((tv.sort_key, block))
        for f in factless:
            entries.append((f.sort_key, f"[fact: {f.text}]"))

        entries.sort(key=lambda e: e[0])
        return "\n\n".join(block for _, block in entries)
