from .activegraph_det import ActiveGraphDetSystem
from .activegraph_sem_extract import ActiveGraphSemExtractSystem
from .activegraph_sem_variants import (
    ActiveGraphSemHybridSystem,
    ActiveGraphSemIndexSystem,
)
from .base import AssembledContext, System
from .full_context_oracle import FullContextOracle
from .full_context_s import FullContextS
from .rag_bm25 import RagBM25
from .rag_dense import RagDense


def build_system(name: str, cfg, *, extract_seed: str = "A") -> System:
    if name == "full-context-oracle":
        return FullContextOracle()
    if name == "full-context-s":
        return FullContextS(
            token_budget=cfg.full_context_s.token_budget,
        )
    if name == "rag-bm25":
        return RagBM25(
            granularity=cfg.retrieval.granularity,
            top_k=cfg.retrieval.top_k,
        )
    if name == "rag-dense":
        return RagDense(
            granularity=cfg.retrieval.granularity,
            top_k=cfg.retrieval.top_k,
            embedding_model=cfg.embeddings.model,
        )
    if name in ("activegraph-det-lexical", "activegraph-det-embedding"):
        signal = "lexical" if name.endswith("-lexical") else "embedding"
        return ActiveGraphDetSystem(
            retrieval_signal=signal,
            token_budget=cfg.activegraph.token_budget,
            min_token_length=cfg.activegraph.min_token_length,
            min_session_cooccurrence=cfg.activegraph.min_session_cooccurrence,
            max_doc_freq_fraction=cfg.activegraph.max_doc_freq_fraction,
            embedding_model=cfg.embeddings.model,
        )
    if name == "activegraph-sem-extract":
        return ActiveGraphSemExtractSystem(
            token_budget=cfg.activegraph.token_budget,
            min_token_length=cfg.activegraph.min_token_length,
            min_session_cooccurrence=cfg.activegraph.min_session_cooccurrence,
            max_doc_freq_fraction=cfg.activegraph.max_doc_freq_fraction,
            extractor_model=cfg.reader.model,
            extract_seed=extract_seed,
        )
    if name == "activegraph-sem-hybrid":
        return ActiveGraphSemHybridSystem(
            token_budget=cfg.activegraph.token_budget,
            min_token_length=cfg.activegraph.min_token_length,
            min_session_cooccurrence=cfg.activegraph.min_session_cooccurrence,
            max_doc_freq_fraction=cfg.activegraph.max_doc_freq_fraction,
            extractor_model=cfg.reader.model,
            extract_seed=extract_seed,
            embedding_model=cfg.embeddings.model,
        )
    if name == "activegraph-sem-index":
        return ActiveGraphSemIndexSystem(
            token_budget=cfg.activegraph.token_budget,
            min_token_length=cfg.activegraph.min_token_length,
            min_session_cooccurrence=cfg.activegraph.min_session_cooccurrence,
            max_doc_freq_fraction=cfg.activegraph.max_doc_freq_fraction,
            extractor_model=cfg.reader.model,
            extract_seed=extract_seed,
            embedding_model=cfg.embeddings.model,
        )
    raise ValueError(f"Unknown system: {name}")


__all__ = [
    "System",
    "AssembledContext",
    "FullContextOracle",
    "FullContextS",
    "RagBM25",
    "RagDense",
    "ActiveGraphDetSystem",
    "ActiveGraphSemExtractSystem",
    "ActiveGraphSemHybridSystem",
    "ActiveGraphSemIndexSystem",
    "build_system",
]
