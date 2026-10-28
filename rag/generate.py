"""Stage 8 - Grounded answer generation with numbered citations.

Backends (picked automatically):
  1. Anthropic API          - if ANTHROPIC_API_KEY is set
  2. OpenAI-compatible API  - if RAG_LLM_BASE_URL is set (e.g. Ollama, vLLM, LM Studio)
  3. Extractive fallback    - no LLM: returns the most relevant sentences, cited
"""
from __future__ import annotations

import logging
import os
import re

from .text import split_sentences, tokenize

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You answer questions using ONLY the numbered sources provided.
Rules:
- Cite every factual claim with its source number in square brackets, e.g. [2] or [1][3].
- If the sources do not contain the answer, say you could not find it in the knowledge base.
- Never invent facts, numbers, or citations.
- Answer in the same language as the question. Be concise."""


def build_context(chunks, max_chars: int = 12000) -> str:
    parts, used = [], 0
    for i, c in enumerate(chunks, start=1):
        m = c.metadata
        header = f"[{i}] {m.get('title', m.get('filename', ''))} (page {m.get('page', 1)})"
        block = f"{header}\n{c.text}"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def _anthropic(question, context, model, max_tokens):
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model, max_tokens=max_tokens, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {question}"}])
    return "".join(b.text for b in msg.content if b.type == "text")


def _openai_compatible(question, context, model, max_tokens):
    import requests
    base = os.environ["RAG_LLM_BASE_URL"].rstrip("/")
    headers = {"Content-Type": "application/json"}
    if os.getenv("RAG_LLM_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['RAG_LLM_API_KEY']}"
    r = requests.post(f"{base}/chat/completions", headers=headers, timeout=120, json={
        "model": model, "max_tokens": max_tokens, "temperature": 0.1,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {question}"}]})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _extractive(question, chunks, max_sentences: int = 4):
    q = set(tokenize(question))
    scored = []
    for i, c in enumerate(chunks, start=1):
        for s in split_sentences(c.text):
            toks = set(tokenize(s))
            if toks:
                overlap = len(q & toks) / (len(q) or 1)
                scored.append((overlap - 0.02 * i, s, i))   # prefer higher-ranked chunks
    scored = [x for x in scored if x[0] > 0]
    if not scored:
        return "I could not find an answer to this in the knowledge base."
    scored.sort(key=lambda x: -x[0])
    return " ".join(f"{s} [{i}]" for _, s, i in scored[:max_sentences])


def generate_answer(question: str, chunks, max_tokens: int = 800) -> tuple[str, str]:
    """Returns (answer_text, backend_name)."""
    if not chunks:
        return "I could not find anything relevant in the knowledge base.", "none"
    context = build_context(chunks)
    try:
        if os.getenv("ANTHROPIC_API_KEY"):
            model = os.getenv("RAG_LLM_MODEL", "claude-sonnet-5-5")
            return _anthropic(question, context, model, max_tokens), f"anthropic:{model}"
        if os.getenv("RAG_LLM_BASE_URL"):
            model = os.getenv("RAG_LLM_MODEL", "llama3.1")
            return _openai_compatible(question, context, model, max_tokens), f"openai-compatible:{model}"
    except Exception as e:
        log.warning("LLM call failed (%s); using extractive fallback.", e)
    return _extractive(question, chunks), "extractive"


def used_citations(answer: str, n: int) -> list[int]:
    nums = {int(x) for x in re.findall(r"\[(\d+)\]", answer)}
    return sorted(x for x in nums if 1 <= x <= n)
