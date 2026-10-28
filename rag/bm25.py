"""Stage 4a - Lexical retrieval: Okapi BM25 over an inverted index (no dependencies)."""
from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict

from .text import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.postings: dict[str, dict[int, int]] = defaultdict(dict)
        self.doc_len: list[int] = []
        self.avgdl = 0.0
        self.idf: dict[str, float] = {}

    def build(self, texts: list[str]):
        self.postings = defaultdict(dict)
        self.doc_len = []
        for i, t in enumerate(texts):
            toks = tokenize(t)
            self.doc_len.append(len(toks))
            for term, c in Counter(toks).items():
                self.postings[term][i] = c
        n = len(self.doc_len)
        self.avgdl = (sum(self.doc_len) / n) if n else 0.0
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5))
                    for t, p in self.postings.items()}
        self.postings = dict(self.postings)

    def search(self, query: str, k: int = 50, allowed: set[int] | None = None):
        scores: dict[int, float] = defaultdict(float)
        k1, b, avgdl = self.k1, self.b, self.avgdl or 1.0
        for term in set(tokenize(query)):
            post = self.postings.get(term)
            if not post:
                continue
            idf = self.idf[term]
            for i, tf in post.items():
                if allowed is not None and i not in allowed:
                    continue
                dl = self.doc_len[i]
                scores[i] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
        return heapq.nlargest(k, scores.items(), key=lambda x: x[1])
