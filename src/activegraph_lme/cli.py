from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import click
from dotenv import load_dotenv
from tqdm import tqdm

from .config import RunCfg, load_config
from .data import LMEInstance, load_dataset, sha256_of_file
from .eval.run_judge import run_judge
from .manifest import Manifest, QueryRecord, repo_sha, submodule_sha
from .reader import AnthropicReader
from .systems import build_system
from .tokens import count_tokens as _tok_count


logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger("aglme")


SYSTEM_PROMPT = (
    "You are a helpful assistant answering a user's question about prior "
    "conversations between the user and an assistant. Use ONLY the provided "
    "conversation history. If the history does not contain enough information "
    "to answer, say you don't know. Be concise."
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
def run_cmd(
    system_name: str,
    dataset_key: str,
    config_path: str,
    smoke: bool,
    limit: int | None,
    run_id: str | None,
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

    reader = AnthropicReader(
        model=cfg.reader.model,
        temperature=cfg.reader.temperature,
        max_tokens=cfg.reader.max_tokens,
    )
    system = build_system(system_name, cfg)

    rid = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = "smoke" if smoke else "full"
    run_dir = Path(cfg.output_dir) / f"{rid}__{system_name}__{dataset_key}__{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

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
        started_at=datetime.now(timezone.utc).isoformat(),
        n_questions=len(instances),
    )
    if system_name == "activegraph":
        manifest.notes.append(
            "ActiveGraphSystem is a STUB (recency-with-budget). Round-two real "
            "internals plug into the same ingest/retrieve interface."
        )

    hyp_path = run_dir / "hypotheses.jsonl"
    t0 = time.monotonic()
    with open(hyp_path, "w") as hyp_f:
        for idx, inst in enumerate(tqdm(instances, desc=f"{system_name}:{dataset_key}")):
            t_inst = time.monotonic()
            state = system.ingest(inst)
            ctx = system.retrieve(state, inst.question, inst.question_date)

            # Determinism check, once per system per run.
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
                    f"({manifest.reader_model_resolved!r} → {out.resolved_model!r}); "
                    f"refusing to record run with ambiguous provenance."
                )

            manifest.queries.append(
                QueryRecord(
                    question_id=inst.question_id,
                    question_type=inst.question_type,
                    context_tokens=_tok_count(ctx.text),
                    prompt_tokens=out.prompt_tokens,
                    completion_tokens=out.completion_tokens,
                    truncated=ctx.truncated,
                    elapsed_s=round(time.monotonic() - t_inst, 4),
                )
            )
            if ctx.truncated:
                manifest.n_truncated += 1

            hyp_f.write(
                json.dumps(
                    {"question_id": inst.question_id, "hypothesis": out.text},
                    ensure_ascii=False,
                )
                + "\n"
            )

    manifest.finished_at = datetime.now(timezone.utc).isoformat()
    manifest.wall_clock_s = round(time.monotonic() - t0, 3)
    manifest.write(run_dir / "manifest.json")
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
