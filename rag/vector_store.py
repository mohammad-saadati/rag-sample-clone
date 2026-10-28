"""Stage 4b - Vector index. Uses FAISS (HNSW) when installed, NumPy otherwise.

Filtering strategy:
  * small allowed set  -> exact search over just those rows (pre-filter)
  * large/no filter    -> ANN search with over-fetch, then post-filter
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


class VectorStore:
    def __init__(self):
        self.vectors: np.ndarray | None = None
        self.index = None

    def build(self, vectors: np.ndarray, use_faiss: bool = True):
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.index = None
        if use_faiss and len(self.vectors) > 5000:
            try:
                import faiss
                idx = faiss.IndexHNSWFlat(self.vectors.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
                idx.hnsw.efConstruction = 200
                idx.hnsw.efSearch = 128
                idx.add(self.vectors)
                self.index = idx
                log.info("Built FAISS HNSW index over %d vectors", len(self.vectors))
            except ImportError:
                log.info("faiss not installed; using exact NumPy search")

    def _exact(self, q, rows, k):
        sims = self.vectors[rows] @ q
        k = min(k, len(rows))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(int(rows[i]), float(sims[i])) for i in top]

    def search(self, q: np.ndarray, k: int = 50, allowed: set[int] | None = None):
        if self.vectors is None or len(self.vectors) == 0:
            return []
        n = len(self.vectors)
        if allowed is not None:
            if not allowed:
                return []
            if self.index is None or len(allowed) < 0.3 * n:
                return self._exact(q, np.fromiter(allowed, dtype=np.int64), k)
        if self.index is not None:
            fetch = k if allowed is None else min(n, k * 4)
            D, I = self.index.search(q[None, :].astype(np.float32), fetch)
            hits = [(int(i), float(d)) for i, d in zip(I[0], D[0])
                    if i >= 0 and (allowed is None or int(i) in allowed)]
            if len(hits) >= min(k, len(allowed) if allowed else k):
                return hits[:k]
            rows = np.fromiter(allowed, dtype=np.int64) if allowed else np.arange(n)
            return self._exact(q, rows, k)
        return self._exact(q, np.arange(n), k)
