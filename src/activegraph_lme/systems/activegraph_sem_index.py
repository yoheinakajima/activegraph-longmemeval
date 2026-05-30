"""``activegraph-sem-index`` — variant (b): facts as retrieval signal ONLY.

Same retrieval signal as ``activegraph-sem-hybrid`` (embedding cosine over
the Fact pool) but the reader NEVER sees a fact. After greedy-selecting
facts by score, each fact is replaced by its ``mentions`` provenance
turn(s); the reader context is just the (de-duplicated, chronological)
turn text — byte-for-byte the same FORMAT det-embedding emits. Only the
*selection path* differs: turns reach the reader because a fact pointed at
them, not because the turn itself embedded close to the question.

This is the cleanest ablation in the set: holding the rendered format
fixed to det-embedding's, it isolates the single question

    "are facts a better retrieval signal than direct turn embedding?"

from the separate question "does putting facts in the reader context
help?" (which ``sem-hybrid`` probes).

Budget accounting — because facts never appear in the rendered context,
the entire 2,500-token budget is spent on turns. We walk facts in score
order, follow ``mentions``, and accumulate UNIQUE turns until the next
turn would exceed budget (a turn's cost is ``tokens(turn.text) + 2`` for
the ``"\\n\\n"`` joiner — identical to retrieve.assemble()'s turn cost).
The chronological render is identical to det-embedding's
``"\\n\\n".join(turn.text ...)``.
"""

from __future__ import annotations

from ..tokens import count_tokens
from .base import AssembledContext
from ._sem_compiled import _FactView, _SemCompiledBase, project_facts


class ActiveGraphSemIndexSystem(_SemCompiledBase):
    """Variant (b): facts index the turns; reader sees turns only."""

    name = "activegraph-sem-index"

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

        selected_turn_ids: list[str] = []
        selected_set: set[str] = set()
        selected_fact_ids: list[str] = []
        n_turns_anchored = 0
        running = 0
        truncated = False

        for f in ranked:
            n_turns_anchored += len(f.turn_ids)
            fact_landed = False
            for tid in f.turn_ids:
                if tid not in turn_by_id:
                    continue
                if tid in selected_set:
                    # Already pulled in by a higher-scored fact — this fact
                    # still counts as "used" (it indexes a rendered turn).
                    fact_landed = True
                    continue
                cost = count_tokens(turn_by_id[tid].text) + 2  # "\n\n" joiner
                if running + cost > self.token_budget:
                    truncated = True
                    continue
                running += cost
                selected_set.add(tid)
                selected_turn_ids.append(tid)
                fact_landed = True
            if fact_landed:
                selected_fact_ids.append(f.fact_id)

        # Chronological, de-duplicated — det-embedding's exact render.
        rendered = sorted(selected_set, key=lambda tid: turn_by_id[tid].sort_key)
        text = "\n\n".join(turn_by_id[tid].text for tid in rendered)

        meta = {
            **state.meta,
            "retrieval_signal": "embedding",
            "assembly": "sem-index",
            "token_budget": self.token_budget,
            "n_facts_selected": len(selected_fact_ids),
            "n_turns_anchored": n_turns_anchored,
            "n_unique_turns_rendered": len(selected_set),
            # Consumed by aic_sidecar.py (turn ids drive AIC hit-rate).
            "selected_turn_ids": rendered,
            "selected_fact_ids": selected_fact_ids,
        }
        return AssembledContext(text=text, truncated=truncated, meta=meta)
