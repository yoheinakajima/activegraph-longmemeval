from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ReaderCfg(BaseModel):
    backend: Literal["anthropic"]
    model: str
    temperature: float = 0.0
    max_tokens: int = 1024


class JudgeCfg(BaseModel):
    short_name: str
    resolved_model: str
    temperature: float = 0.0


class EmbeddingsCfg(BaseModel):
    backend: Literal["openai"]
    model: str


class RetrievalCfg(BaseModel):
    granularity: Literal["turn", "session"] = "session"
    top_k: int = 10


class FullContextSCfg(BaseModel):
    token_budget: int = 180_000
    truncation: Literal["oldest_first"] = "oldest_first"


class RunCfg(BaseModel):
    reader: ReaderCfg
    judge: JudgeCfg
    embeddings: EmbeddingsCfg
    retrieval: RetrievalCfg
    full_context_s: FullContextSCfg
    systems: list[str]
    datasets: dict[str, str]
    seed: int = 42
    output_dir: str = "runs"


def load_config(path: str | Path = "config/run.yaml") -> RunCfg:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return RunCfg.model_validate(raw)
