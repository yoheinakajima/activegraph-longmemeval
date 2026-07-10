from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
from dotenv import load_dotenv
from tqdm import tqdm

from .config import RunCfg, load_config
from .data import LMEInstance, load_dataset, sha256_of_file
from .eval.run_judge import run_judge
from .manifest import Manifest, QueryRecord, repo_sha, submodule_sha
from .reader import AnthropicReader
from .systems import build_system
from .tokens import count_tokens as _tok_count, token_source


logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger("aglme")


SYSTEM_PROMPT = (
    "You are a helpful assistant answering a user's question about prior "
    "conversations between the user and an assistant. Use ONLY the provided "
    "conversation history. If the history does not contain enough information "
    "to answer, say you don't know. The history may begin with a "
    "[compiled-memory] proof packet. Proof completion means required evidence "
    "fields are present; it does not guarantee answer correctness. Check every "
    "candidate against its cited rows and raw sources. For approximate "
    "relative-time questions, apply the packet's stated tolerance rather than "
    "preferring calendar-day equality. Be concise."
)


def format_user(context: str, question: str, question_date: str) -> str:
    return (
        f"Conversation history:\n{context}\n\n"
        f"Today's date: {question_date}\n"
        f"Question: {question}\n"
        f"Answer:"
    )


def _filter_smoke(instances: list[LMEInstance], smoke_ids_path: Path) -> list[LMEInstance]:
    if not smoke_ids_path.exists():
        raise FileNotFoundError(
            f"--smoke requires {smoke_ids_path}. Run `make smoke-ids` after `make data` "
            f"to generate (and commit) the frozen stratified subset."
        )
    ids = {
        line.strip()
        for line in smoke_ids_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    selected = [i for i in instances if i.question_id in ids]
    missing = ids - {i.question_id for i in selected}
    if missing:
        raise RuntimeError(
            f"smoke_ids.txt references question_ids not in dataset: "
            f"{sorted(list(missing))[:5]}... ({len(missing)} total)."
        )
    return selected


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    tmp.replace(path)


def _append_run_event(path: Path, event_type: str, payload: dict[str, Any]) -> None:
    row = {
        "event_id": str(uuid.uuid4()),
        "type": event_type,
        "created_at": _utc_now(),
        "payload": payload,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _read_query_records(path: Path) -> list[QueryRecord]:
    if not path.exists():
        return []
    records: dict[str, QueryRecord] = {}
    order: list[str] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            qid = str(row["question_id"])
            rec = QueryRecord(**row)
        except Exception as e:  # noqa: BLE001
            raise click.ClickException(
                f"Cannot parse {path}:{line_no} as a QueryRecord: {e}"
            ) from e
        if qid not in records:
            order.append(qid)
        records[qid] = rec
    return [records[qid] for qid in order]


def _repair_hypotheses_for_resume(
    hyp_path: Path,
    completed_records: list[QueryRecord],
) -> None:
    if not completed_records:
        return
    if not hyp_path.exists():
        raise click.ClickException(
            f"Cannot resume: {hyp_path} is missing but query_records.jsonl has "
            f"{len(completed_records)} completed records."
        )
    by_qid: dict[str, dict[str, Any]] = {}
    for line_no, line in enumerate(hyp_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            qid = str(row["question_id"])
        except Exception as e:  # noqa: BLE001
            raise click.ClickException(
                f"Cannot parse {hyp_path}:{line_no} while preparing resume: {e}"
            ) from e
        by_qid[qid] = row

    missing = [rec.question_id for rec in completed_records if rec.question_id not in by_qid]
    if missing:
        raise click.ClickException(
            "Cannot resume: completed query_records are missing hypotheses for "
            f"{missing[:5]} ({len(missing)} total)."
        )

    tmp = hyp_path.with_suffix(hyp_path.suffix + ".resume.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in completed_records:
            f.write(json.dumps(by_qid[rec.question_id], ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(hyp_path)


def _embedding_cache_manifest(system: Any) -> dict[str, Any]:
    embedder = getattr(system, "_embedder", None)
    output = (
        embedder.cache_stats()
        if embedder is not None and hasattr(embedder, "cache_stats")
        else {}
    )
    vector_store = getattr(system, "_memory_vector_store", None)
    if vector_store is not None and hasattr(vector_store, "stats"):
        output["activegraph_memory_vector_store"] = vector_store.stats()
        output["activegraph_memory_vector_store_path"] = str(
            getattr(system, "memory_embedding_cache", "")
        )
    return output


@click.group()
def main() -> None:
    """activegraph-longmemeval harness."""
    load_dotenv()


@main.command("run")
@click.option("--system", "system_name", required=True, type=str)
@click.option("--dataset", "dataset_key", required=True, type=click.Choice(["oracle", "s"]))
@click.option("--config", "config_path", default="config/run.yaml", show_default=True)
@click.option("--smoke", is_flag=True, help="Use the frozen 50-question subset.")
@click.option("--limit", type=int, default=None, help="Cap to first N (debug only).")
@click.option("--run-id", type=str, default=None, help="Override run id (default: timestamp).")
@click.option(
    "--resume",
    is_flag=True,
    help=(
        "Resume an existing run directory by skipping question_ids already "
        "present in query_records.jsonl."
    ),
)
@click.option(
    "--require-authoritative-tokens/--allow-charfallback",
    "require_auth_tokens",
    default=None,
    help=(
        "If set, fail the run when context_tokens would be recorded with the "
        "char/4 fallback instead of tiktoken. Default: on for full runs, "
        "off (warn-only) for --smoke."
    ),
)
@click.option(
    "--extract-seed",
    type=click.Choice(["A", "A-v2", "B", "C"]),
    default="A-v2",
    show_default=True,
    help=(
        "Frozen extraction-cache seed for the sem-extract family. seed-A is "
        "the original user-only cache (INVALIDATED under the role-aware "
        "extractor — its manifest guard will refuse to load). seed-A-v2 is "
        "the canonical role-aware cache (user + assistant facts); seed-B/C "
        "are gitignored variance samples. Other systems ignore this flag."
    ),
)
def run_cmd(
    system_name: str,
    dataset_key: str,
    config_path: str,
    smoke: bool,
    limit: int | None,
    run_id: str | None,
    resume: bool,
    require_auth_tokens: bool | None,
    extract_seed: str,
) -> None:
    cfg = load_config(config_path)
    dataset_path = Path(cfg.datasets[dataset_key])
    if not dataset_path.exists():
        raise click.ClickException(
            f"Dataset not found: {dataset_path}. Run `make data` first."
        )

    instances = load_dataset(dataset_path)
    if smoke:
        instances = _filter_smoke(instances, Path("config/smoke_ids.txt"))
    if limit is not None:
        instances = instances[:limit]

    # Resolve the token-counting source up front so the gate fires before we
    # spend any API budget.
    ctx_src = token_source()
    if require_auth_tokens is None:
        require_auth_tokens = not smoke  # default: ON for full, OFF (warn) for smoke
    if ctx_src != "tiktoken":
        msg = (
            f"context_token_source resolved to {ctx_src!r}; tiktoken did not "
            f"load. Authoritative reader (prompt/completion) tokens are still "
            f"recorded from the API, but the cross-system context yardstick is "
            f"approximated (char/4). To fix permanently, run any single command "
            f"with network so tiktoken populates $TIKTOKEN_CACHE_DIR "
            f"({os.environ.get('TIKTOKEN_CACHE_DIR')!r})."
        )
        if require_auth_tokens:
            raise click.ClickException(msg + " Failing per --require-authoritative-tokens.")
        log.warning(msg)

    reader = AnthropicReader(
        model=cfg.reader.model,
        temperature=cfg.reader.temperature,
        max_tokens=cfg.reader.max_tokens,
    )
    system = build_system(system_name, cfg, extract_seed=extract_seed)

    rid = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = "smoke" if smoke else "full"
    run_dir = Path(cfg.output_dir) / f"{rid}__{system_name}__{dataset_key}__{tag}"
    run_dir_existed = run_dir.exists()
    run_dir.mkdir(parents=True, exist_ok=True)
    hyp_path = run_dir / "hypotheses.jsonl"
    query_records_path = run_dir / "query_records.jsonl"
    retrieval_records_path = run_dir / "retrieval_records.jsonl"
    event_path = run_dir / "run_events.jsonl"
    state_path = run_dir / "run_state.json"
    partial_manifest_path = run_dir / "manifest.partial.json"
    final_manifest_path = run_dir / "manifest.json"

    artifacts = [hyp_path, query_records_path, final_manifest_path, partial_manifest_path]
    if cfg.activegraph.save_retrieval_artifacts:
        artifacts.append(retrieval_records_path)
    if not resume and any(p.exists() for p in artifacts):
        raise click.ClickException(
            f"Run directory already contains artifacts: {run_dir}. "
            "Pass --resume to continue it or choose a new --run-id."
        )

    resume_state: dict[str, Any] = {}
    if resume and state_path.exists():
        resume_state = json.loads(state_path.read_text())
    elif resume:
        prior_manifest_path = (
            partial_manifest_path
            if partial_manifest_path.exists()
            else final_manifest_path
            if final_manifest_path.exists()
            else None
        )
        if prior_manifest_path is not None:
            prior = json.loads(prior_manifest_path.read_text())
            resume_state = {
                "started_at": prior.get("started_at"),
                "reader_model_resolved": prior.get("reader_model_resolved"),
                "accumulated_wall_clock_s": prior.get("wall_clock_s", 0.0),
                "embedding_cache": prior.get("embedding_cache", {}),
                "system_identity": prior.get("system_identity", {}),
            }

    completed_records = _read_query_records(query_records_path) if resume else []
    instance_ids = [inst.question_id for inst in instances]
    unknown_completed = sorted({r.question_id for r in completed_records} - set(instance_ids))
    if unknown_completed:
        raise click.ClickException(
            "Cannot resume: query_records.jsonl contains question_ids outside "
            f"this run selection: {unknown_completed[:5]} ({len(unknown_completed)} total)."
        )
    completed_by_id = {r.question_id: r for r in completed_records}
    completed_records = [completed_by_id[qid] for qid in instance_ids if qid in completed_by_id]
    completed_ids = set(completed_by_id)
    if resume:
        _repair_hypotheses_for_resume(hyp_path, completed_records)

    started_at = str(resume_state.get("started_at") or _utc_now())
    previous_wall_clock_s = float(resume_state.get("accumulated_wall_clock_s") or 0.0)

    manifest = Manifest(
        run_id=run_dir.name,
        system=system_name,
        dataset_path=str(dataset_path),
        dataset_sha256=sha256_of_file(dataset_path),
        config=json.loads(cfg.model_dump_json()),
        repo_sha=repo_sha(),
        submodule_sha=submodule_sha(),
        reader_model_requested=cfg.reader.model,
        reader_model_resolved="",  # filled by first API response
        judge_short_name=cfg.judge.short_name,
        judge_resolved_model=cfg.judge.resolved_model,
        seed=cfg.seed,
        started_at=started_at,
        n_questions=len(instances),
        context_token_source=ctx_src,
        require_authoritative_tokens=require_auth_tokens,
    )
    manifest.reader_model_resolved = str(resume_state.get("reader_model_resolved") or "")
    manifest.embedding_cache = dict(resume_state.get("embedding_cache") or {})
    manifest.system_identity = dict(resume_state.get("system_identity") or {})
    manifest.queries = list(completed_records)
    manifest.n_truncated = sum(1 for q in manifest.queries if q.truncated)
    if system_name.startswith("activegraph-det-"):
        signal = system_name.removeprefix("activegraph-det-")
        manifest.notes.append(
            f"ActiveGraph deterministic Mode A; retrieval_signal={signal}. "
            f"No LLM extraction at ingest. Token budget mirrors the turn-level "
            f"RAG baselines so accuracy comparisons aren't confounded by context size."
        )
    if resume:
        manifest.notes.append(
            f"Resumable run: loaded {len(completed_records)} completed query records "
            f"from {query_records_path.name}."
        )

    if not run_dir_existed or not resume:
        _append_run_event(
            event_path,
            "run.started",
            {
                "run_id": run_dir.name,
                "system": system_name,
                "dataset": dataset_key,
                "n_questions": len(instances),
            },
        )
    else:
        _append_run_event(
            event_path,
            "run.resumed",
            {
                "run_id": run_dir.name,
                "system": system_name,
                "dataset": dataset_key,
                "completed_questions": len(completed_records),
                "remaining_questions": len(instances) - len(completed_records),
            },
        )

    remaining = [inst for inst in instances if inst.question_id not in completed_ids]
    t0 = time.monotonic()
    try:
        with open(hyp_path, "a" if resume else "w", encoding="utf-8") as hyp_f, open(
            query_records_path, "a" if resume else "w", encoding="utf-8"
        ) as qrec_f, open(
            retrieval_records_path if cfg.activegraph.save_retrieval_artifacts else os.devnull,
            "a" if resume else "w",
            encoding="utf-8",
        ) as retrieval_f:
            for idx, inst in enumerate(
                tqdm(remaining, desc=f"{system_name}:{dataset_key}")
            ):
                _append_run_event(
                    event_path,
                    "query.started",
                    {
                        "question_id": inst.question_id,
                        "question_type": inst.question_type,
                        "remaining_index": idx,
                        "completed_before_query": len(manifest.queries),
                    },
                )
                t_inst = time.monotonic()
                state = system.ingest(inst)
                ctx = system.retrieve(state, inst.question, inst.question_date)

                # Determinism check, once per process invocation.
                if idx == 0:
                    ctx2 = system.retrieve(state, inst.question, inst.question_date)
                    if ctx.text != ctx2.text:
                        raise RuntimeError(
                            f"System {system_name}.retrieve() is non-deterministic "
                            f"under fixed state — refusing to record run."
                        )

                user = format_user(ctx.text, inst.question, inst.question_date)
                out = reader.generate(SYSTEM_PROMPT, user)

                # Pin the resolved reader model on the first successful call.
                if not manifest.reader_model_resolved:
                    manifest.reader_model_resolved = out.resolved_model
                elif manifest.reader_model_resolved != out.resolved_model:
                    raise RuntimeError(
                        "Reader resolved model changed mid-run "
                        f"({manifest.reader_model_resolved!r} -> {out.resolved_model!r}); "
                        f"refusing to record run with ambiguous provenance."
                    )

                context_meta = ctx.meta or {}
                identity = {
                    key: context_meta.get(key)
                    for key in (
                        "activegraph_memory_version",
                        "activegraph_memory_content_hash",
                        "activegraph_memory_git_commit",
                        "activegraph_memory_git_dirty",
                    )
                    if key in context_meta
                }
                if identity:
                    if manifest.system_identity and manifest.system_identity != identity:
                        raise RuntimeError(
                            "External system identity changed mid-run "
                            f"({manifest.system_identity!r} -> {identity!r})."
                        )
                    manifest.system_identity = identity
                pipeline = context_meta.get("pipeline_telemetry") or {}
                compiled = context_meta.get("compiled_evidence") or {}
                record = QueryRecord(
                    question_id=inst.question_id,
                    question_type=inst.question_type,
                    context_tokens=_tok_count(ctx.text),
                    prompt_tokens=out.prompt_tokens,
                    completion_tokens=out.completion_tokens,
                    truncated=ctx.truncated,
                    elapsed_s=round(time.monotonic() - t_inst, 4),
                    retrieval_latency_ms=float(pipeline.get("duration_ms") or 0.0),
                    retrieval_cost_usd=float(pipeline.get("cost_usd") or 0.0),
                    runtime_profile=str(context_meta.get("memory_profile") or ""),
                    proof_complete=(
                        bool(compiled.get("proof_complete")) if compiled else None
                    ),
                )
                hyp_f.write(
                    json.dumps(
                        {"question_id": inst.question_id, "hypothesis": out.text},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                hyp_f.flush()
                os.fsync(hyp_f.fileno())

                qrec_f.write(json.dumps(record.__dict__, ensure_ascii=False) + "\n")
                qrec_f.flush()
                os.fsync(qrec_f.fileno())

                if cfg.activegraph.save_retrieval_artifacts:
                    retrieval_f.write(
                        json.dumps(
                            {
                                "question_id": inst.question_id,
                                "question_type": inst.question_type,
                                "question": inst.question,
                                "question_date": inst.question_date,
                                "context_text": ctx.text,
                                "context_tokens": record.context_tokens,
                                "truncated": ctx.truncated,
                                "meta": context_meta,
                            },
                            ensure_ascii=False,
                            default=str,
                        )
                        + "\n"
                    )
                    retrieval_f.flush()
                    os.fsync(retrieval_f.fileno())

                manifest.queries.append(record)
                if ctx.truncated:
                    manifest.n_truncated += 1
                manifest.embedding_cache = (
                    _embedding_cache_manifest(system) or manifest.embedding_cache
                )
                manifest.wall_clock_s = round(
                    previous_wall_clock_s + (time.monotonic() - t0), 3
                )
                state_payload = {
                    "run_id": run_dir.name,
                    "started_at": manifest.started_at,
                    "updated_at": _utc_now(),
                    "reader_model_resolved": manifest.reader_model_resolved,
                    "accumulated_wall_clock_s": manifest.wall_clock_s,
                    "completed_question_ids": [q.question_id for q in manifest.queries],
                    "n_completed": len(manifest.queries),
                    "n_questions": len(instances),
                    "query_records_path": str(query_records_path),
                    "hypotheses_path": str(hyp_path),
                    "partial_manifest_path": str(partial_manifest_path),
                    "retrieval_records_path": str(retrieval_records_path),
                    "embedding_cache": manifest.embedding_cache,
                    "system_identity": manifest.system_identity,
                }
                _atomic_write_json(state_path, state_payload)
                manifest.write(partial_manifest_path)
                _append_run_event(
                    event_path,
                    "query.completed",
                    {
                        "question_id": inst.question_id,
                        "question_type": inst.question_type,
                        "completed_questions": len(manifest.queries),
                        "n_questions": len(instances),
                        "elapsed_s": record.elapsed_s,
                    },
                )
    except Exception as e:
        manifest.embedding_cache = _embedding_cache_manifest(system) or manifest.embedding_cache
        manifest.wall_clock_s = round(previous_wall_clock_s + (time.monotonic() - t0), 3)
        manifest.write(partial_manifest_path)
        _append_run_event(
            event_path,
            "run.failed",
            {
                "run_id": run_dir.name,
                "error_type": e.__class__.__name__,
                "error": str(e),
                "completed_questions": len(manifest.queries),
                "n_questions": len(instances),
            },
        )
        raise

    if not manifest.reader_model_resolved:
        raise click.ClickException(
            "Run has no resolved reader model. This can only happen if all "
            "questions were marked completed before the resumable state file "
            "recorded reader_model_resolved."
        )

    manifest.finished_at = _utc_now()
    manifest.wall_clock_s = round(previous_wall_clock_s + (time.monotonic() - t0), 3)
    manifest.embedding_cache = _embedding_cache_manifest(system) or manifest.embedding_cache
    manifest.write(final_manifest_path)
    _atomic_write_json(
        state_path,
        {
            "run_id": run_dir.name,
            "started_at": manifest.started_at,
            "finished_at": manifest.finished_at,
            "updated_at": _utc_now(),
            "reader_model_resolved": manifest.reader_model_resolved,
            "accumulated_wall_clock_s": manifest.wall_clock_s,
            "completed_question_ids": [q.question_id for q in manifest.queries],
            "n_completed": len(manifest.queries),
            "n_questions": len(instances),
            "query_records_path": str(query_records_path),
            "hypotheses_path": str(hyp_path),
            "manifest_path": str(final_manifest_path),
            "retrieval_records_path": str(retrieval_records_path),
            "embedding_cache": manifest.embedding_cache,
            "system_identity": manifest.system_identity,
        },
    )
    _append_run_event(
        event_path,
        "run.completed",
        {
            "run_id": run_dir.name,
            "completed_questions": len(manifest.queries),
            "n_questions": len(instances),
            "wall_clock_s": manifest.wall_clock_s,
        },
    )
    click.echo(str(run_dir))


@main.command("eval")
@click.option("--run-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--config", "config_path", default="config/run.yaml", show_default=True)
def eval_cmd(run_dir: str, config_path: str) -> None:
    cfg = load_config(config_path)
    manifest = json.loads(Path(run_dir, "manifest.json").read_text())

    # Pick the reference file per dataset automatically, never silently mismatch.
    dataset_path = manifest["dataset_path"]
    if dataset_path.endswith("longmemeval_oracle.json"):
        reference_path = cfg.datasets["oracle"]
    elif dataset_path.endswith("longmemeval_s_cleaned.json"):
        reference_path = cfg.datasets["s"]
    else:
        raise click.ClickException(
            f"Cannot auto-select reference for dataset_path={dataset_path!r}"
        )

    scores = run_judge(run_dir, reference_path, judge_short_name=cfg.judge.short_name)
    click.echo(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
