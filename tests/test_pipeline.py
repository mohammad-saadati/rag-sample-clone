"""Run with:  python -m pytest -q"""
from pathlib import Path

import pytest

from rag import KBConfig, KnowledgeBase
from rag.filters import MetadataIndex, matches
from rag.hybrid import rrf
from rag.text import split_sentences

SAMPLE = Path(__file__).parent.parent / "data" / "sample"


@pytest.fixture(scope="module")
def kb(tmp_path_factory):
    kb = KnowledgeBase(tmp_path_factory.mktemp("kb"),
                       KBConfig(embedding_model="hashing-1024", reranker_model="heuristic"))
    assert kb.ingest([SAMPLE]) > 0
    kb.save()
    return KnowledgeBase.load(kb.dir)


def test_retrieves_right_document(kb):
    ans = kb.query("What does error code E17 mean?")
    assert ans.sources[0]["title"] == "atlas mower manual"
    assert "[1]" in ans.answer


def test_metadata_filter_restricts_results(kb):
    ans = kb.query("battery", filters={"type": "pdf"})
    assert ans.sources and all(s["source"].endswith(".pdf") for s in ans.sources)


def test_empty_filter_result(kb):
    assert kb.query("battery", filters={"department": "nope"}).sources == []


def test_persian_query(kb):
    ans = kb.query("گارانتی باتری چند سال است؟")
    assert ans.sources[0]["title"] == "warranty fa"


def test_graph_finds_entities(kb):
    assert "VoltaCell" in kb.graph.match_query("who is voltacell")


def test_rrf_rewards_agreement():
    fused = rrf({"a": [(1, 9), (2, 8)], "b": [(2, 5), (3, 4)]})
    assert fused[0][0] == 2


def test_filter_operators():
    m = {"year": 2025, "tags": ["policy", "hr"], "type": "pdf"}
    assert matches(m, {"year": {"$gte": 2024}, "tags": {"$contains": "hr"}})
    assert not matches(m, {"$or": [{"type": "txt"}, {"year": {"$lt": 2000}}]})
    idx = MetadataIndex(); idx.build([m, {"type": "txt", "year": 2020}])
    assert idx.allowed({"type": {"$in": ["pdf"]}, "year": {"$gt": 2021}}) == {0}


def test_abbreviation_not_split():
    assert len(split_sentences("It was led by Dr. Lena Okafor. Done.")) == 2
