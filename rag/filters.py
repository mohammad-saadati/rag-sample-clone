"""Stage 3 - Metadata filtering (Mongo-style) with an inverted index for fast equality.

Examples:
    {"type": "pdf"}
    {"year": {"$gte": 2022}, "department": {"$in": ["legal", "hr"]}}
    {"$or": [{"type": "pdf"}, {"tags": {"$contains": "policy"}}]}
"""
from __future__ import annotations

from collections import defaultdict


def _cmp(value, op, target) -> bool:
    try:
        if op == "$eq":
            return value == target
        if op == "$ne":
            return value != target
        if op == "$gt":
            return value is not None and value > target
        if op == "$gte":
            return value is not None and value >= target
        if op == "$lt":
            return value is not None and value < target
        if op == "$lte":
            return value is not None and value <= target
        if op == "$in":
            if isinstance(value, list):
                return any(v in target for v in value)
            return value in target
        if op == "$nin":
            return value not in target
        if op == "$contains":
            if isinstance(value, (list, tuple, set)):
                return target in value
            return isinstance(value, str) and str(target).lower() in value.lower()
        if op == "$exists":
            return (value is not None) == bool(target)
    except TypeError:
        return False
    raise ValueError(f"Unknown filter operator {op}")


def matches(meta: dict, flt: dict) -> bool:
    for key, cond in flt.items():
        if key == "$and":
            if not all(matches(meta, f) for f in cond):
                return False
        elif key == "$or":
            if not any(matches(meta, f) for f in cond):
                return False
        elif key == "$not":
            if matches(meta, cond):
                return False
        else:
            value = meta.get(key)
            if isinstance(cond, dict):
                if not all(_cmp(value, op, t) for op, t in cond.items()):
                    return False
            elif isinstance(value, list):
                if cond not in value:
                    return False
            elif value != cond:
                return False
    return True


class MetadataIndex:
    """value -> set(row ids) for low-cardinality fields; falls back to scanning."""

    INDEXED = ("type", "year", "department", "language", "tags", "doc_id", "filename")

    def __init__(self):
        self.metas: list[dict] = []
        self.inv: dict[str, dict] = defaultdict(lambda: defaultdict(set))

    def build(self, metas: list[dict]):
        self.metas = metas
        self.inv = defaultdict(lambda: defaultdict(set))
        for i, m in enumerate(metas):
            for f in self.INDEXED:
                v = m.get(f)
                for item in (v if isinstance(v, list) else [v]):
                    if item is not None:
                        self.inv[f][item].add(i)

    def allowed(self, flt: dict | None) -> set[int] | None:
        """Row ids satisfying `flt`, or None meaning 'everything'."""
        if not flt:
            return None
        candidate: set[int] | None = None
        rest = {}
        for key, cond in flt.items():
            if key in self.inv and not isinstance(cond, dict):
                ids = self.inv[key].get(cond, set())
            elif key in self.inv and isinstance(cond, dict) and set(cond) == {"$in"}:
                ids = set().union(*(self.inv[key].get(v, set()) for v in cond["$in"])) \
                    if cond["$in"] else set()
            else:
                rest[key] = cond
                continue
            candidate = ids if candidate is None else candidate & ids
        pool = range(len(self.metas)) if candidate is None else candidate
        if not rest:
            return set(pool)
        return {i for i in pool if matches(self.metas[i], rest)}
