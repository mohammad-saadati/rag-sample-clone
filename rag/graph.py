"""Stage 7 - GraphRAG: an entity co-occurrence graph built from the chunks.

Nodes are entities (people, organisations, products, acronyms...), edges link
entities mentioned in the same chunk (weighted by count). At query time the
entities in the question are matched, expanded to their strongest neighbours,
and chunks mentioning those entities are returned as an extra ranked list that
feeds into RRF. This surfaces multi-hop evidence that neither BM25 nor vectors
would rank highly on their own.

Extractors:
  * RegexEntityExtractor (default, offline): capitalised phrases, acronyms,
    quoted terms and hashtag-style terms; works for English, plus Persian
    terms listed in a user-supplied glossary.
  * LLMEntityExtractor (optional): asks Claude for entities in any language.
"""
from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter, defaultdict
from itertools import combinations

import networkx as nx

from .text import STOPWORDS, normalize, tokenize

log = logging.getLogger(__name__)

_CAPS = re.compile(r"\b(?:[A-Z][a-zA-Z0-9&\-]+(?:\s+(?:of|de|and|&)?\s*[A-Z][a-zA-Z0-9&\-]+){0,4})\b")
_ACRONYM = re.compile(r"\b[A-Z]{2,}[0-9]*\b")
_QUOTED = re.compile(r"[\"«“]([^\"»”]{3,40})[\"»”]")
_SENT_START_NOISE = {"The", "This", "That", "These", "Those", "In", "On", "At", "For", "A", "An",
                     "It", "We", "Our", "If", "When", "All", "Each", "After", "Before", "As",
                     "Its", "Their", "His", "Her", "Page", "Step", "Note", "Section", "Chapter"}


class RegexEntityExtractor:
    def __init__(self, glossary: list[str] | None = None):
        self.glossary = [normalize(g) for g in (glossary or [])]

    def __call__(self, text: str) -> set[str]:
        ents = set()
        for m in _CAPS.finditer(text):
            words = m.group(0).split()
            before = text[:m.start()].rstrip()
            at_sentence_start = not before or before[-1] in ".!?:\n#*-"
            while words and words[0] in _SENT_START_NOISE:
                words = words[1:]
            # a lone capitalised word opening a sentence is usually just grammar ("Error", "Check")
            if at_sentence_start and len(words) == 1 and m.group(0).split()[0] == words[0]:
                continue
            name = " ".join(words)
            if len(name) >= 3 and name.lower() not in STOPWORDS:
                ents.add(name)
        ents.update(_ACRONYM.findall(text))
        ents.update(q.strip() for q in _QUOTED.findall(text))
        low = text.lower()
        ents.update(g for g in self.glossary if g.lower() in low)
        return ents


class LLMEntityExtractor:
    """Uses the Anthropic API. Slow/costly for 100k docs - use for high-value subsets."""

    def __init__(self, model: str | None = None):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model or os.getenv("RAG_LLM_MODEL", "claude-sonnet-5-5")

    def __call__(self, text: str) -> set[str]:
        msg = self.client.messages.create(
            model=self.model, max_tokens=300,
            system="Extract named entities (people, organisations, products, places, "
                   "projects, laws, technical terms). Reply ONLY with a JSON array of strings.",
            messages=[{"role": "user", "content": text[:4000]}])
        raw = "".join(b.text for b in msg.content if b.type == "text")
        try:
            return {str(e).strip() for e in json.loads(raw.strip("` \njson")) if str(e).strip()}
        except json.JSONDecodeError:
            return set()


def _key(name: str) -> str:
    return " ".join(tokenize(name, keep_stopwords=True))


class EntityGraph:
    def __init__(self, extractor=None):
        self.extractor = extractor or RegexEntityExtractor()
        self.g = nx.Graph()
        self.entity_rows: dict[str, set[int]] = defaultdict(set)
        self.lookup: dict[str, str] = {}      # normalised key -> node name
        self.max_ngram = 1

    def build(self, texts: list[str], min_mentions: int = 1):
        per_row = [self.extractor(t) for t in texts]
        counts = Counter(e for ents in per_row for e in ents)
        keep = {e for e, c in counts.items() if c >= min_mentions}
        self.g = nx.Graph()
        self.entity_rows = defaultdict(set)
        for row, ents in enumerate(per_row):
            ents = sorted(e for e in ents if e in keep)
            for e in ents:
                self.entity_rows[e].add(row)
                if not self.g.has_node(e):
                    self.g.add_node(e)
            for a, b in combinations(ents, 2):
                w = self.g[a][b]["weight"] + 1 if self.g.has_edge(a, b) else 1
                self.g.add_edge(a, b, weight=w)
        for n in self.g.nodes:
            self.g.nodes[n]["mentions"] = len(self.entity_rows[n])
        self.lookup = {_key(n): n for n in self.g.nodes if _key(n)}
        self.max_ngram = max((len(k.split()) for k in self.lookup), default=1)
        log.info("Entity graph: %d nodes, %d edges", self.g.number_of_nodes(),
                 self.g.number_of_edges())

    def match_query(self, query: str) -> list[str]:
        toks = tokenize(query, keep_stopwords=True)
        found = []
        for n in range(min(self.max_ngram, len(toks)), 0, -1):
            for i in range(len(toks) - n + 1):
                node = self.lookup.get(" ".join(toks[i:i + n]))
                if node and node not in found:
                    found.append(node)
        return found

    def search(self, query: str, k: int = 30, hops: int = 1, max_neighbors: int = 5,
               allowed: set[int] | None = None):
        seeds = self.match_query(query)
        if not seeds:
            return [], {"seeds": [], "expanded": []}
        weights = {s: 1.0 for s in seeds}
        frontier = list(seeds)
        for depth in range(1, hops + 1):
            nxt = []
            for node in frontier:
                nbrs = sorted(self.g[node].items(), key=lambda x: -x[1]["weight"])[:max_neighbors]
                for nb, data in nbrs:
                    w = 0.5 ** depth * min(1.0, data["weight"] / 3)
                    if w > weights.get(nb, 0):
                        weights[nb] = w
                        nxt.append(nb)
            frontier = nxt
        scores = defaultdict(float)
        for ent, w in weights.items():
            rows = self.entity_rows.get(ent, ())
            idf = 1.0 / (1 + len(rows)) ** 0.5    # rare entities count more
            for r in rows:
                if allowed is None or r in allowed:
                    scores[r] += w * idf
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:k]
        expanded = [e for e in weights if e not in seeds]
        return ranked, {"seeds": seeds, "expanded": expanded}

    def neighborhood(self, entity: str, limit: int = 10):
        if entity not in self.g:
            return []
        return sorted(((nb, d["weight"]) for nb, d in self.g[entity].items()),
                      key=lambda x: -x[1])[:limit]
