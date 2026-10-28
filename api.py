"""HTTP API.   uvicorn api:app --reload      (KB_DIR env var selects the store)

POST /ask     {"question": "...", "filters": {...}, "k": 6}
GET  /stats
"""
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag import KnowledgeBase

app = FastAPI(title="RAG knowledge base")
kb = KnowledgeBase.load(os.getenv("KB_DIR", "kb_store"))


class AskRequest(BaseModel):
    question: str
    filters: dict | None = None
    k: int | None = None


@app.post("/ask")
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(400, "question is empty")
    return kb.query(req.question, filters=req.filters, final_k=req.k).to_dict()


@app.get("/stats")
def stats():
    return kb.stats()
