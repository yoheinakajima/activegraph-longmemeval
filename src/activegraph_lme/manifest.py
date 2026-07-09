from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _git(*args: str, cwd: str | Path = ".") -> str:
    return subprocess.check_output(["git", *args], cwd=str(cwd), text=True).strip()


def repo_sha() -> str:
    return _git("rev-parse", "HEAD")


def submodule_sha(submodule_path: str = "third_party/longmemeval") -> str:
    return _git("rev-parse", "HEAD", cwd=submodule_path)


@dataclass
class QueryRecord:
    question_id: str
    question_type: str
    context_tokens: int       # tiktoken estimate of assembled context only
    prompt_tokens: int        # authoritative count from reader API (system + user)
    completion_tokens: int    # authoritative count from reader API
    truncated: bool = False   # True iff context was truncated to fit a budget
    elapsed_s: float = 0.0


@dataclass
class Manifest:
    run_id: str
    system: str
    dataset_path: str
    dataset_sha256: str
    config: dict[str, Any]
    repo_sha: str
    submodule_sha: str
    reader_model_requested: str
    reader_model_resolved: str   # MUST be populated from first API response; never empty
    judge_short_name: str
    judge_resolved_model: str
    seed: int
    python_version: str = field(default_factory=platform.python_version)
    started_at: str = ""
    finished_at: str = ""
    wall_clock_s: float = 0.0
    n_questions: int = 0
    n_truncated: int = 0
    # Authoritative for prompt/completion (always "api" — reader.usage); for
    # context_tokens it is "tiktoken" or "charfallback".
    prompt_completion_token_source: str = "api"
    context_token_source: str = ""        # filled at run start; never empty in a valid run
    require_authoritative_tokens: bool = True
    embedding_cache: dict[str, Any] = field(default_factory=dict)
    queries: list[QueryRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "run_id": self.run_id,
            "system": self.system,
            "dataset_path": self.dataset_path,
            "dataset_sha256": self.dataset_sha256,
            "config": self.config,
            "repo_sha": self.repo_sha,
            "submodule_sha": self.submodule_sha,
            "reader_model_requested": self.reader_model_requested,
            "reader_model_resolved": self.reader_model_resolved,
            "judge_short_name": self.judge_short_name,
            "judge_resolved_model": self.judge_resolved_model,
            "seed": self.seed,
            "python_version": self.python_version,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "wall_clock_s": self.wall_clock_s,
            "n_questions": self.n_questions,
            "n_truncated": self.n_truncated,
            "prompt_completion_token_source": self.prompt_completion_token_source,
            "context_token_source": self.context_token_source,
            "require_authoritative_tokens": self.require_authoritative_tokens,
            "embedding_cache": self.embedding_cache,
            "queries": [q.__dict__ for q in self.queries],
            "notes": self.notes,
        }
        return d

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
