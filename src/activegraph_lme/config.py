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


class ActiveGraphCfg(BaseModel):
    # Mirrors the turn-level RAG baselines so accuracy comparisons aren't
    # confounded by context size (see paper/results_tables.md: rag-dense/turn
    # on `s` is mean 2398 context tokens).
    token_budget: int = 2500
    # Corpus-relative thresholds. NOT tuned hyperparameters — they're rules
    # that pin the lexical vocabulary deterministically. Changing them shifts
    # the entire signal across the benchmark, which is fine but MUST be
    # reported, not hidden.
    min_token_length: int = 4
    min_session_cooccurrence: int = 2
    max_doc_freq_fraction: float = 0.5
    # Radius (in turns) for the global session-date temporal expansion that
    # runs alongside the intra-session 1-hop pairing. A seed turn pulls in
    # its N nearest neighbors in the global (session_date, session_idx,
    # turn_idx) ordering, so consecutive-day evidence on temporal-reasoning
    # questions survives the budget even when its raw similarity is lower
    # than unrelated higher-scoring turns. Default 2 is the smallest value
    # that reliably crosses a session boundary for the common 2-turn
    # user/assistant session: it reaches the paired turn AND the first turn
    # of the adjacent session by date. Set to 0 to disable cross-session
    # expansion (intra-session pairing is preserved regardless).
    temporal_expansion_hops: int = 2


class RunCfg(BaseModel):
    reader: ReaderCfg
    judge: JudgeCfg
    embeddings: EmbeddingsCfg
    retrieval: RetrievalCfg
    full_context_s: FullContextSCfg
    activegraph: ActiveGraphCfg = Field(default_factory=ActiveGraphCfg)
    systems: list[str]
    datasets: dict[str, str]
    seed: int = 42
    output_dir: str = "runs"


def load_config(path: str | Path = "config/run.yaml") -> RunCfg:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return RunCfg.model_validate(raw)
