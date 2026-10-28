"""The KnowledgeBase ties every stage together.

Indexing path:  files -> Documents -> Chunks -> {BM25, vectors, entity graph, metadata index}
Query path:     question (+filters)
                  -> metadata pre-filter
                  -> BM25 top-N  |  vector top-N  |  graph top-N      (in parallel lists)
                  -> RRF fusion -> rerank top-K -> LLM with numbered sources -> cited answer
"""
from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .bm25 import BM25Index
from .chunking import Chunk, chunk_document
from .embeddings import get_embedder
from .filters import MetadataIndex
from .generate import generate_answer, used_citations
from .graph import EntityGraph, LLMEntityExtractor, RegexEntityExtractor
from .hybrid import rrf
from .ingest import load_documents
from .rerank import get_reranker, rerank
from .vector_store import VectorStore

log = logging.getLogger(__name__)


@dataclass
class KBConfig:
    embedding_model: str = "intfloat/multilingual-e5-base"   # or "hashing-1024" offline
    reranker_model: str | None = "BAAI/bge-reranker-v2-m3"   # or "heuristic" / None
    chunk_words: int = 200
    chunk_overlap: int = 40
    bm25_k: int = 50           # candidates from each retriever
    dense_k: int = 50
    graph_k: int = 30
    rrf_k: int = 60
    rrf_weights: dict = field(default_factory=lambda: {"bm25": 1.0, "dense": 1.0, "graph": 0.7})
    rerank_pool: int = 30      # how many fused candidates the reranker sees
    final_k: int = 6           # chunks sent to the LLM
    use_graph: bool = True
    graph_extractor: str = "regex"            # "regex" | "llm"
    graph_glossary: list = field(default_factory=list)
    ingest_workers: int = 1
    ocr_lang: str = "eng+fas"


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[dict]
    backend: str
    timings: dict
    debug: dict

    def to_dict(self):
        return asdict(self)


class KnowledgeBase:
    def __init__(self, store_dir: str | Path, config: KBConfig | None = None):
        self.dir = Path(store_dir)
        self.cfg = config or KBConfig()
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None
        self.bm25 = BM25Index()
        self.vs = VectorStore()
        self.meta_index = MetadataIndex()
        self.graph: EntityGraph | None = None
        self._embedder = None
        self._reranker = None

    # ---------- lazy models ----------
    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = get_embedder(self.cfg.embedding_model)
        return self._embedder

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = get_reranker(self.cfg.reranker_model, self.embedder)
        return self._reranker

    # ---------- indexing ----------
    def ingest(self, paths, batch_size: int = 256) -> int:
        docs = load_documents(paths, workers=self.cfg.ingest_workers, ocr_lang=self.cfg.ocr_lang)
        known = {c.doc_id for c in self.chunks}
        new_chunks = []
        for d in docs:
            if d.doc_id in known:
                log.info("Re-ingesting %s: replacing old chunks", d.metadata["filename"])
                self._drop_doc(d.doc_id)
            new_chunks.extend(chunk_document(d, self.cfg.chunk_words, self.cfg.chunk_overlap))
        if not new_chunks:
            return 0

        t = time.time()
        texts = [c.indexed_text for c in new_chunks]
        vecs = np.vstack([self.embedder.embed_passages(texts[i:i + batch_size])
                          for i in range(0, len(texts), batch_size)])
        log.info("Embedded %d chunks in %.1fs", len(texts), time.time() - t)
        self.chunks.extend(new_chunks)
        self.embeddings = vecs if self.embeddings is None else np.vstack([self.embeddings, vecs])
        self._build_indexes()
        return len(new_chunks)

    def _drop_doc(self, doc_id: str):
        keep = [i for i, c in enumerate(self.chunks) if c.doc_id != doc_id]
        self.chunks = [self.chunks[i] for i in keep]
        if self.embeddings is not None:
            self.embeddings = self.embeddings[keep]

    def _build_indexes(self):
        texts = [c.indexed_text for c in self.chunks]
        for c in self.chunks:
            c.metadata["doc_id"] = c.doc_id
        self.bm25.build(texts)
        self.vs.build(self.embeddings)
        self.meta_index.build([c.metadata for c in self.chunks])
        if self.cfg.use_graph:
            extractor = (LLMEntityExtractor() if self.cfg.graph_extractor == "llm"
                         else RegexEntityExtractor(self.cfg.graph_glossary))
            self.graph = EntityGraph(extractor)
            self.graph.build([c.text for c in self.chunks])

    # ---------- persistence ----------
    def save(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        np.save(self.dir / "embeddings.npy", self.embeddings)
        with open(self.dir / "indexes.pkl", "wb") as f:
            graph_state = None
            if self.graph:
                graph_state = (self.graph.g, dict(self.graph.entity_rows),
                               self.graph.lookup, self.graph.max_ngram)
            pickle.dump({"bm25": self.bm25, "graph": graph_state}, f)
        cfg = asdict(self.cfg)
        cfg["embedding_model_resolved"] = self.embedder.name
        (self.dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
        log.info("Saved knowledge base to %s", self.dir)

    @classmethod
    def load(cls, store_dir: str | Path, **overrides) -> "KnowledgeBase":
        store_dir = Path(store_dir)
        raw = json.loads((store_dir / "config.json").read_text())
        resolved = raw.pop("embedding_model_resolved", None)
        raw.update(overrides)
        if resolved and "embedding_model" not in overrides:
            raw["embedding_model"] = resolved   # must query with the same embedder
        kb = cls(store_dir, KBConfig(**raw))
        with open(store_dir / "chunks.jsonl", encoding="utf-8") as f:
            kb.chunks = [Chunk.from_dict(json.loads(line)) for line in f]
        kb.embeddings = np.load(store_dir / "embeddings.npy")
        with open(store_dir / "indexes.pkl", "rb") as f:
            state = pickle.load(f)
        kb.bm25 = state["bm25"]
        kb.vs.build(kb.embeddings)
        kb.meta_index.build([c.metadata for c in kb.chunks])
        if state["graph"]:
            kb.graph = EntityGraph()
            kb.graph.g, rows, kb.graph.lookup, kb.graph.max_ngram = state["graph"]
            kb.graph.entity_rows.update(rows)
        return kb

    # ---------- retrieval ----------
    def retrieve(self, question: str, filters: dict | None = None, final_k: int | None = None):
        cfg, timings, debug = self.cfg, {}, {}
        final_k = final_k or cfg.final_k

        t = time.time()
        allowed = self.meta_index.allowed(filters)
        timings["filter_ms"] = (time.time() - t) * 1000
        debug["allowed_chunks"] = len(self.chunks) if allowed is None else len(allowed)
        if allowed is not None and not allowed:
            return [], timings, debug

        t = time.time()
        lists = {"bm25": self.bm25.search(question, cfg.bm25_k, allowed)}
        timings["bm25_ms"] = (time.time() - t) * 1000

        t = time.time()
        qv = self.embedder.embed_query(question)
        lists["dense"] = self.vs.search(qv, cfg.dense_k, allowed)
        timings["dense_ms"] = (time.time() - t) * 1000

        if self.graph is not None and cfg.use_graph:
            t = time.time()
            g_hits, g_info = self.graph.search(question, cfg.graph_k, allowed=allowed)
            lists["graph"] = g_hits
            debug["graph"] = g_info
            timings["graph_ms"] = (time.time() - t) * 1000

        t = time.time()
        fused = rrf(lists, k=cfg.rrf_k, weights=cfg.rrf_weights, top_n=cfg.rerank_pool)
        timings["rrf_ms"] = (time.time() - t) * 1000

        t = time.time()
        rows = [r for r, _, _ in fused]
        reranked = rerank(self.reranker, question, rows,
                          [self.chunks[r].indexed_text for r in rows], final_k)
        timings["rerank_ms"] = (time.time() - t) * 1000

        fused_info = {r: (s, ranks) for r, s, ranks in fused}
        results = []
        for row, score in reranked:
            c = self.chunks[row]
            results.append({"chunk": c, "rerank_score": round(score, 4),
                            "rrf_score": round(fused_info[row][0], 5),
                            "ranks": fused_info[row][1]})
        debug["candidates"] = {k: len(v) for k, v in lists.items()}
        debug["fused"] = len(fused)
        return results, timings, debug

    def query(self, question: str, filters: dict | None = None,
              final_k: int | None = None) -> Answer:
        t0 = time.time()
        results, timings, debug = self.retrieve(question, filters, final_k)
        t = time.time()
        chunks = [r["chunk"] for r in results]
        text, backend = generate_answer(question, chunks)
        timings["generate_ms"] = (time.time() - t) * 1000
        timings["total_ms"] = (time.time() - t0) * 1000
        cited = set(used_citations(text, len(chunks)))
        sources = []
        for i, r in enumerate(results, start=1):
            m = r["chunk"].metadata
            sources.append({
                "n": i, "cited": i in cited, "title": m.get("title"), "source": m.get("source"),
                "page": m.get("page"), "chunk_id": r["chunk"].chunk_id,
                "rerank_score": r["rerank_score"], "rrf_score": r["rrf_score"],
                "ranks": r["ranks"], "snippet": r["chunk"].text[:300]})
        timings = {k: round(v, 1) for k, v in timings.items()}
        return Answer(question, text, sources, backend, timings, debug)

    def stats(self) -> dict:
        return {
            "documents": len({c.doc_id for c in self.chunks}),
            "chunks": len(self.chunks),
            "vocab": len(self.bm25.idf),
            "embedding_model": self.embedder.name,
            "embedding_dim": int(self.embeddings.shape[1]) if self.embeddings is not None else 0,
            "graph_nodes": self.graph.g.number_of_nodes() if self.graph else 0,
            "graph_edges": self.graph.g.number_of_edges() if self.graph else 0,
        }
