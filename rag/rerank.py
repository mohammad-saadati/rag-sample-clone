"""Stage 6 - Reranking the fused shortlist.

Default: a multilingual cross-encoder (BAAI/bge-reranker-v2-m3) that reads the
query and each chunk *together*. Fallback: a heuristic mixing embedding cosine,
query-term coverage and term proximity.
"""
from __future__ import annotations

import logging

import numpy as np

from .text import tokenize

log = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, max_length=512)
        self.name = model_name

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return [float(s) for s in self.model.predict([(query, t) for t in texts])]


class HeuristicReranker:
    name = "heuristic"

    def __init__(self, embedder):
        self.embedder = embedder

    @staticmethod
    def _coverage_and_proximity(q_terms: set[str], text: str) -> tuple[float, float]:
        toks = tokenize(text)
        if not q_terms or not toks:
            return 0.0, 0.0
        present = q_terms & set(toks)
        coverage = len(present) / len(q_terms)
        if len(present) < 2:
            return coverage, coverage
        # smallest window containing every present query term
        need, have, best, left = len(present), {}, len(toks), 0
        for right, t in enumerate(toks):
            if t in present:
                have[t] = have.get(t, 0) + 1
            while len(have) == need:
                best = min(best, right - left + 1)
                lt = toks[left]
                if lt in have:
                    have[lt] -= 1
                    if have[lt] == 0:
                        del have[lt]
                left += 1
        return coverage, need / best

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        q = self.embedder.embed_query(query)
        cos = self.embedder.embed_passages(texts) @ q
        q_terms = set(tokenize(query))
        out = []
        for c, t in zip(cos, texts):
            cov, prox = self._coverage_and_proximity(q_terms, t)
            out.append(float(0.5 * c + 0.35 * cov + 0.15 * prox))
        return out


def get_reranker(name: str | None, embedder):
    if name and name != "heuristic":
        try:
            return CrossEncoderReranker(name)
        except Exception as e:
            log.warning("Reranker '%s' unavailable (%s); using heuristic.", name, e)
    return HeuristicReranker(embedder)


def rerank(reranker, query: str, rows: list[int], texts: list[str], top_k: int):
    scores = reranker.score(query, texts)
    order = np.argsort(scores)[::-1][:top_k]
    return [(rows[i], scores[i]) for i in order]
