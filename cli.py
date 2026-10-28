#!/usr/bin/env python
"""Command-line interface.

  python cli.py ingest data/sample --kb kb_store
  python cli.py ask "What is the refund window?" --kb kb_store
  python cli.py ask "..." --kb kb_store --filter '{"type": "pdf", "year": {"$gte": 2024}}' --debug
  python cli.py stats --kb kb_store
  python cli.py entity "Northwind" --kb kb_store
"""
import argparse
import json
import logging
import sys
from pathlib import Path

from rag import KBConfig, KnowledgeBase


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true")
    p = argparse.ArgumentParser(description="Hybrid RAG knowledge base")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="index files or folders", parents=[common])
    pi.add_argument("paths", nargs="+")
    pi.add_argument("--kb", default="kb_store")
    pi.add_argument("--embedding-model", default=None,
                    help="e.g. intfloat/multilingual-e5-base or hashing-1024")
    pi.add_argument("--reranker", default=None, help="cross-encoder name or 'heuristic'")
    pi.add_argument("--workers", type=int, default=1)
    pi.add_argument("--no-graph", action="store_true")
    pi.add_argument("--glossary", default=None, help="text file, one entity per line")

    pa = sub.add_parser("ask", help="ask a question", parents=[common])
    pa.add_argument("question")
    pa.add_argument("--kb", default="kb_store")
    pa.add_argument("--filter", default=None, help="JSON metadata filter")
    pa.add_argument("--k", type=int, default=None)
    pa.add_argument("--debug", action="store_true")
    pa.add_argument("--json", action="store_true")

    ps = sub.add_parser("stats", parents=[common]); ps.add_argument("--kb", default="kb_store")
    pe = sub.add_parser("entity", help="show an entity's graph neighbourhood", parents=[common])
    pe.add_argument("name"); pe.add_argument("--kb", default="kb_store")

    args = p.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    if args.cmd == "ingest":
        kb_dir = Path(args.kb)
        if (kb_dir / "config.json").exists():
            kb = KnowledgeBase.load(kb_dir)
        else:
            cfg = KBConfig(ingest_workers=args.workers, use_graph=not args.no_graph)
            if args.embedding_model:
                cfg.embedding_model = args.embedding_model
            if args.reranker:
                cfg.reranker_model = args.reranker
            if args.glossary:
                cfg.graph_glossary = [l.strip() for l in open(args.glossary, encoding="utf-8") if l.strip()]
            kb = KnowledgeBase(kb_dir, cfg)
        n = kb.ingest(args.paths)
        kb.save()
        print(f"Indexed {n} new chunks.")
        print(json.dumps(kb.stats(), indent=2, ensure_ascii=False))

    elif args.cmd == "ask":
        kb = KnowledgeBase.load(args.kb)
        flt = json.loads(args.filter) if args.filter else None
        ans = kb.query(args.question, filters=flt, final_k=args.k)
        if args.json:
            print(json.dumps(ans.to_dict(), indent=2, ensure_ascii=False, default=str))
            return
        print("\n" + ans.answer + "\n")
        print("Sources:")
        for s in ans.sources:
            mark = "*" if s["cited"] else " "
            print(f" {mark}[{s['n']}] {s['title']} (p.{s['page']})  {s['source']}")
        if args.debug:
            print("\nRetriever ranks per source:")
            for s in ans.sources:
                print(f"  [{s['n']}] ranks={s['ranks']} rrf={s['rrf_score']} rerank={s['rerank_score']}")
            print("Debug:", json.dumps(ans.debug, ensure_ascii=False))
            print("Timings (ms):", ans.timings, "| backend:", ans.backend)

    elif args.cmd == "stats":
        print(json.dumps(KnowledgeBase.load(args.kb).stats(), indent=2, ensure_ascii=False))

    elif args.cmd == "entity":
        kb = KnowledgeBase.load(args.kb)
        if not kb.graph:
            sys.exit("This knowledge base was built without a graph.")
        matches = kb.graph.match_query(args.name) or [args.name]
        for m in matches:
            print(f"{m}: {kb.graph.neighborhood(m)}")


if __name__ == "__main__":
    main()
