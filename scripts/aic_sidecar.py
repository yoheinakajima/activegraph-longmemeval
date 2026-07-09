"""Replay ONLY retrieval for an already-completed run, no reader/LLM call.

Reads <run_dir>/manifest.json to reconstruct the exact RunCfg, the exact
dataset, and the exact set of question_ids that were judged. Writes
<run_dir>/aic_sidecar.jsonl with {question_id, selected_turn_ids,
selected_session_ids, system, granularity} for downstream answer-in-context
scoring.

Determinism: matches cli.py's repeat-call check; for ActiveGraph and RAG it
also asserts the reconstructed context text equals system.retrieve(...).text
so the sidecar's selection is provably the same projection the published run
fed the reader.

Cost: zero Anthropic calls. The embedding signal still issues OpenAI
embedding requests (same ones the published run made; OPENAI_API_KEY
required for activegraph-det-embedding and rag-dense).

Usage:
    python scripts/aic_sidecar.py <run_dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm

from activegraph_lme.config import RunCfg
from activegraph_lme.data import LMEInstance, load_dataset, sha256_of_file
from activegraph_lme.systems import build_system
from activegraph_lme.systems.activegraph_det import ActiveGraphDetSystem
from activegraph_lme.systems.activegraph_sem_extract import (
    ActiveGraphSemExtractSystem,
    _score_units_lexical,
)
from activegraph_lme.systems.rag_bm25 import RagBM25
from activegraph_lme.systems.rag_dense import RagDense
from activegraph_lme.activegraph.retrieve import (
    assemble,
    score_embedding,
    score_lexical,
)


def _partition_unit_ids(unit_ids: list[str]) -> tuple[list[str], list[str]]:
    """Split a selected_unit_ids list into (turn_ids, fact_ids).

    Turn ids carry the ``{session_id}#{turn_idx}`` shape; fact ids use
    the ``fact:<sha256-prefix>`` shape and contain no ``#``. Anything
    else is treated as a turn id (preserves pre-Stage-1 behavior for
    legacy callers).
    """
    turn_ids = [uid for uid in unit_ids if "#" in uid and not uid.startswith("fact:")]
    fact_ids = [uid for uid in unit_ids if uid.startswith("fact:")]
    return turn_ids, fact_ids


def _ag_select(system: ActiveGraphDetSystem, state, question: str) -> tuple[list[str], str]:
    if system.retrieval_signal == "lexical":
        scores = score_lexical(
            state.state, question, min_token_length=system.min_token_length
        )
    else:
        embedder = system._get_embedder()
        scores, _ = score_embedding(
            state.state, question, embedder, turn_embeddings=state.turn_embeddings
        )
    res = assemble(state.state, scores, token_budget=system.token_budget)
    return res.selected_unit_ids, res.text


def _ag_sem_select(
    system: ActiveGraphSemExtractSystem, state, question: str
) -> tuple[list[str], str]:
    """Mirror of _ag_select for the sem-extract system.

    Reuses the system's own _score_units_lexical so facts and turns are
    scored on the same yardstick the live retrieve() uses; the byte-
    identical reconstruction assertion in main() then catches any drift
    between this path and ActiveGraphSemExtractSystem.retrieve().
    """
    scores = _score_units_lexical(
        state.state, question, min_token_length=system.min_token_length
    )
    res = assemble(state.state, scores, token_budget=system.token_budget)
    return res.selected_unit_ids, res.text


def _rag_dense_select(
    system: RagDense, state, question: str, inst: LMEInstance
) -> tuple[list[tuple[str, int | None]], str]:
    from activegraph_lme.systems.rag_dense import _embed_batch

    ids = _enumerate_docs(inst, system.granularity)
    q_emb = _embed_batch(system.embedding_model, [question])[0]
    sims = state.embeddings @ q_emb
    ranked = sorted(
        range(len(state.docs)),
        key=lambda i: (-float(sims[i]), state.docs[i].sort_key),
    )
    picked = ranked[: state.top_k]
    picked.sort(key=lambda i: state.docs[i].sort_key)
    selected = [ids[i] for i in picked]
    text = "\n\n".join(state.docs[i].text for i in picked)
    return selected, text


def _rag_bm25_select(
    system: RagBM25, state, question: str, inst: LMEInstance
) -> tuple[list[tuple[str, int | None]], str]:
    from activegraph_lme.systems.rag_bm25 import _tokenize

    ids = _enumerate_docs(inst, system.granularity)
    scores = state.bm25.get_scores(_tokenize(question))
    ranked = sorted(
        range(len(state.docs)),
        key=lambda i: (-float(scores[i]), state.docs[i].sort_key),
    )
    picked = ranked[: state.top_k]
    picked.sort(key=lambda i: state.docs[i].sort_key)
    selected = [ids[i] for i in picked]
    text = "\n\n".join(state.docs[i].text for i in picked)
    return selected, text


def _enumerate_docs(inst: LMEInstance, granularity: str) -> list[tuple[str, int | None]]:
    """Mirror the (sid, t_idx) order produced by RagBM25/RagDense.ingest()."""
    out: list[tuple[str, int | None]] = []
    for sid, _date, turns in zip(
        inst.haystack_session_ids, inst.haystack_dates, inst.haystack_sessions
    ):
        if granularity == "session":
            out.append((sid, None))
        else:
            for t_idx in range(len(turns)):
                out.append((sid, t_idx))
    return out


def _full_context_select(inst: LMEInstance) -> list[tuple[str, int | None]]:
    out: list[tuple[str, int | None]] = []
    for sid, _date, turns in zip(
        inst.haystack_session_ids, inst.haystack_dates, inst.haystack_sessions
    ):
        for t_idx in range(len(turns)):
            out.append((sid, t_idx))
    return out


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=str)
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Override sidecar output path (default: <run_dir>/aic_sidecar.jsonl).",
    )
    ap.add_argument(
        "--skip-dataset-sha-check",
        action="store_true",
        help="Skip sha256 verification of dataset against manifest (not recommended).",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    system_name: str = manifest["system"]
    dataset_path = Path(manifest["dataset_path"])

    if not dataset_path.exists():
        raise SystemExit(
            f"Dataset not found at {dataset_path} (recorded in manifest). "
            f"Run `make data` or copy the file in."
        )
    if not args.skip_dataset_sha_check:
        actual = sha256_of_file(dataset_path)
        if actual != manifest["dataset_sha256"]:
            raise SystemExit(
                f"Dataset sha256 mismatch: on disk {actual}, manifest pins "
                f"{manifest['dataset_sha256']}. Refusing to replay against the "
                f"wrong corpus."
            )

    cfg = RunCfg.model_validate(manifest["config"])
    instances_all = load_dataset(dataset_path)
    judged_ids = [q["question_id"] for q in manifest["queries"]]
    judged_set = set(judged_ids)
    instances = [i for i in instances_all if i.question_id in judged_set]
    missing = judged_set - {i.question_id for i in instances}
    if missing:
        raise SystemExit(
            f"manifest references {len(missing)} question_ids not in dataset "
            f"(e.g. {sorted(missing)[:3]}). Aborting."
        )
    by_qid = {i.question_id: i for i in instances}
    ordered = [by_qid[qid] for qid in judged_ids]

    system = build_system(system_name, cfg)
    out_path = Path(args.out) if args.out else (run_dir / "aic_sidecar.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    granularity = cfg.retrieval.granularity if system_name.startswith("rag-") else "turn"

    n_full = 0
    with open(out_path, "w") as f:
        for idx, inst in enumerate(
            tqdm(ordered, desc=f"sidecar:{system_name}:{run_dir.name[:30]}")
        ):
            state = system.ingest(inst)

            selected_fact_ids: list[str] = []
            if system_name.startswith("activegraph-det-"):
                selected_unit_ids, sel_text = _ag_select(system, state, inst.question)
                if idx == 0:
                    _ids2, _text2 = _ag_select(system, state, inst.question)
                    if _ids2 != selected_unit_ids:
                        raise SystemExit(
                            f"Non-deterministic ActiveGraph selection on "
                            f"{inst.question_id}: refusing to record."
                        )
                ctx = system.retrieve(state, inst.question, inst.question_date)
                if ctx.text != sel_text:
                    raise SystemExit(
                        f"Sidecar reconstruction != system.retrieve for "
                        f"{inst.question_id}; harness drift, refusing to record."
                    )
                selected_turn_ids, selected_fact_ids = _partition_unit_ids(selected_unit_ids)
                # Derive session ids from TURN ids only — fact ids
                # (fact:<hash>, no `#`) would otherwise mis-split and
                # pollute the per-question session set.
                selected_session_ids = sorted({tid.rsplit("#", 1)[0] for tid in selected_turn_ids})

            elif system_name == "activegraph-sem-extract":
                selected_unit_ids, sel_text = _ag_sem_select(system, state, inst.question)
                # No idx==0 determinism check: ingest is LLM-driven so
                # selection across two ingest() calls would differ. The
                # repeat-call check on retrieve() (below) still holds
                # because retrieve over a frozen state is deterministic.
                ctx = system.retrieve(state, inst.question, inst.question_date)
                if ctx.text != sel_text:
                    raise SystemExit(
                        f"Sidecar reconstruction != system.retrieve for "
                        f"{inst.question_id}; harness drift, refusing to record."
                    )
                ctx2 = system.retrieve(state, inst.question, inst.question_date)
                if ctx2.text != ctx.text:
                    raise SystemExit(
                        f"Non-deterministic sem-extract retrieve on "
                        f"{inst.question_id}; refusing to record."
                    )
                selected_turn_ids, selected_fact_ids = _partition_unit_ids(selected_unit_ids)
                selected_session_ids = sorted({tid.rsplit("#", 1)[0] for tid in selected_turn_ids})

            elif system_name in (
                "activegraph-sem-hybrid",
                "activegraph-sem-index",
                "activegraph-memory-pack",
            ):
                # Compiled-memory systems expose their selected turn / fact
                # ids directly on retrieve().meta, so we read those
                # rather than re-deriving selection. Assembly is pure given
                # the (content-addressed, cached) embedding scores, so a
                # repeat retrieve() is byte-identical — assert it.
                ctx = system.retrieve(state, inst.question, inst.question_date)
                ctx2 = system.retrieve(state, inst.question, inst.question_date)
                if ctx2.text != ctx.text:
                    raise SystemExit(
                        f"Non-deterministic {system_name} retrieve on "
                        f"{inst.question_id}; refusing to record."
                    )
                meta = ctx.meta or {}
                selected_turn_ids = list(meta.get("selected_turn_ids", []))
                selected_fact_ids = list(meta.get("selected_fact_ids", []))
                selected_session_ids = sorted(
                    {tid.rsplit("#", 1)[0] for tid in selected_turn_ids}
                )

            elif system_name in ("rag-bm25", "rag-dense"):
                if system_name == "rag-dense":
                    sel_pairs, sel_text = _rag_dense_select(system, state, inst.question, inst)
                else:
                    sel_pairs, sel_text = _rag_bm25_select(system, state, inst.question, inst)
                if idx == 0:
                    if system_name == "rag-dense":
                        _p2, _t2 = _rag_dense_select(system, state, inst.question, inst)
                    else:
                        _p2, _t2 = _rag_bm25_select(system, state, inst.question, inst)
                    if _p2 != sel_pairs:
                        raise SystemExit(
                            f"Non-deterministic {system_name} selection on "
                            f"{inst.question_id}; refusing to record."
                        )
                ctx = system.retrieve(state, inst.question, inst.question_date)
                if ctx.text != sel_text:
                    raise SystemExit(
                        f"Sidecar reconstruction != system.retrieve for "
                        f"{inst.question_id}; harness drift, refusing to record."
                    )
                if granularity == "turn":
                    selected_turn_ids = [f"{sid}#{tidx}" for sid, tidx in sel_pairs]
                else:
                    selected_turn_ids = []
                selected_session_ids = sorted({sid for sid, _ in sel_pairs})

            elif system_name.startswith("full-context"):
                # Trivially every haystack turn is "selected" by the reader.
                pairs = _full_context_select(inst)
                selected_turn_ids = [f"{sid}#{tidx}" for sid, tidx in pairs]
                selected_session_ids = sorted({sid for sid, _ in pairs})
                n_full += 1

            else:
                raise SystemExit(f"Unknown system: {system_name}")

            record = {
                "question_id": inst.question_id,
                "question_type": inst.question_type,
                "system": system_name,
                "granularity": granularity,
                "selected_turn_ids": selected_turn_ids,
                "selected_session_ids": selected_session_ids,
            }
            # Additive field: emitted only when the system actually
            # selected fact units. This keeps pre-Stage-1 sidecar output
            # (det-*, rag-*, full-context) byte-identical to the pre-patch
            # baseline; only sem-extract records grow the new field.
            if selected_fact_ids:
                record["selected_fact_ids"] = selected_fact_ids
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    print(f"[sidecar] wrote {out_path} ({len(ordered)} questions, "
          f"{n_full} full-context)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
