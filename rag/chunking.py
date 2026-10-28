"""Stage 2 - Chunking: sentence-aware, overlapping windows that never cross pages.

Each chunk carries its document's metadata plus page and position, and a short
contextual header (document title) that is prepended for embedding/BM25.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .ingest import Document
from .text import split_sentences


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def indexed_text(self) -> str:
        """Text used for indexing: title header + body (helps short chunks)."""
        title = self.metadata.get("title", "")
        return f"{title}\n{self.text}" if title else self.text

    def to_dict(self):
        return {"chunk_id": self.chunk_id, "doc_id": self.doc_id,
                "text": self.text, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def _wc(s: str) -> int:
    return len(s.split())


def chunk_document(doc: Document, max_words: int = 200, overlap_words: int = 40,
                   min_words: int = 15) -> list[Chunk]:
    chunks: list[Chunk] = []
    idx = 0
    for page_no, page in enumerate(doc.pages, start=1):
        sents = split_sentences(page)
        # break very long "sentences" (tables, OCR garbage) into word windows
        expanded = []
        for s in sents:
            words = s.split()
            if len(words) > max_words:
                for i in range(0, len(words), max_words - overlap_words):
                    expanded.append(" ".join(words[i:i + max_words]))
            else:
                expanded.append(s)

        window: list[str] = []
        for s in expanded:
            if window and _wc(" ".join(window)) + _wc(s) > max_words:
                chunks.append(_make(doc, window, page_no, idx)); idx += 1
                # carry the tail sentences over as overlap
                tail, n = [], 0
                for prev in reversed(window):
                    n += _wc(prev)
                    if n > overlap_words:
                        break
                    tail.insert(0, prev)
                window = tail
            window.append(s)
        if window:
            if chunks and _wc(" ".join(window)) < min_words and \
                    chunks[-1].metadata["page"] == page_no:
                chunks[-1].text += " " + " ".join(window)   # merge a tiny remainder
            else:
                chunks.append(_make(doc, window, page_no, idx)); idx += 1
    return chunks


def _make(doc: Document, sents: list[str], page: int, idx: int) -> Chunk:
    meta = {k: v for k, v in doc.metadata.items() if k != "content_hash"}
    meta.update({"page": page, "chunk_index": idx})
    return Chunk(chunk_id=f"{doc.doc_id}:{idx}", doc_id=doc.doc_id,
                 text=" ".join(sents).strip(), metadata=meta)
