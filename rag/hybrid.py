"""Stage 5 - Hybrid fusion with Reciprocal Rank Fusion (RRF).

RRF(d) = sum_i  w_i / (k + rank_i(d))   -- scale-free, so BM25 and cosine scores
never need calibrating against each other.
"""
from __future__ import annotations

from collections import defaultdict


def rrf(result_lists: dict[str, list[tuple[int, float]]], k: int = 60,
        weights: dict[str, float] | None = None, top_n: int | None = None):
    weights = weights or {}
    fused: dict[int, float] = defaultdict(float)
    ranks: dict[int, dict[str, int]] = defaultdict(dict)
    for name, results in result_lists.items():
        w = weights.get(name, 1.0)
        for rank, (row, _score) in enumerate(results, start=1):
            fused[row] += w / (k + rank)
            ranks[row][name] = rank
    ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    if top_n:
        ordered = ordered[:top_n]
    return [(row, score, ranks[row]) for row, score in ordered]
