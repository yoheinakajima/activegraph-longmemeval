from .base import System, AssembledContext
from .full_context_oracle import FullContextOracle
from .full_context_s import FullContextS
from .rag_bm25 import RagBM25
from .rag_dense import RagDense
from .activegraph_stub import ActiveGraphSystem


def build_system(name: str, cfg) -> System:
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
    if name == "activegraph":
        return ActiveGraphSystem(token_budget=cfg.full_context_s.token_budget)
    raise ValueError(f"Unknown system: {name}")


__all__ = [
    "System",
    "AssembledContext",
    "FullContextOracle",
    "FullContextS",
    "RagBM25",
    "RagDense",
    "ActiveGraphSystem",
    "build_system",
]
