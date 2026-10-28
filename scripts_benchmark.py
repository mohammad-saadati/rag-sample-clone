"""Synthetic scale test: python scripts_benchmark.py 100000
Generates N short synthetic documents in memory, indexes them, and times queries."""
import random
import sys
import time

from rag import KBConfig, KnowledgeBase
from rag.chunking import chunk_document
from rag.ingest import Document

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
random.seed(0)
products = [f"Model{chr(65 + i % 26)}{i}" for i in range(500)]
words = ("battery motor firmware warranty refund sensor dock blade wheel update error "
         "signal charge safety recall supplier invoice shipping policy manual").split()
kb = KnowledgeBase("bench_store", KBConfig(embedding_model="hashing-512", reranker_model="heuristic"))
t = time.time()
chunks = []
for i in range(N):
    p = random.choice(products)
    body = f"{p} " + " ".join(random.choices(words, k=60)) + f". Error code E{i % 900} affects {p}."
    d = Document(f"d{i}", [body], {"title": f"doc {i}", "type": random.choice(["pdf", "txt"]),
                                    "year": 2018 + i % 8, "source": f"synthetic/{i}"})
    chunks.extend(chunk_document(d))
kb.chunks = chunks
kb.embeddings = kb.embedder.embed_passages([c.indexed_text for c in chunks])
kb._build_indexes()
print(f"indexed {len(chunks)} chunks in {time.time() - t:.1f}s")
for q, f in [("error code E421 firmware", None), ("battery recall ModelC2", {"year": {"$gte": 2024}})]:
    t = time.time(); res, tm, _ = kb.retrieve(q, f)
    print(f"{q!r} filter={f}: {(time.time() - t) * 1000:.0f} ms  top={res[0]['chunk'].metadata['title']}  {tm}")
