"""rag: a hybrid (BM25 + dense + GraphRAG) retrieval-augmented generation knowledge base."""
from .pipeline import KnowledgeBase, KBConfig

__all__ = ["KnowledgeBase", "KBConfig"]
