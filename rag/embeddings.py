"""Stage 4b - Dense embeddings.

Default model: intfloat/multilingual-e5-base (good for Persian + English).
If sentence-transformers or the model is unavailable, a deterministic hashing
embedder is used so the pipeline still runs end-to-end (quality is lower).
"""
from __future__ import annotations

import logging
import zlib

import numpy as np

from .text import tokenize

log = logging.getLogger(__name__)


class HashingEmbedder:
    """Feature-hashed word + char-trigram vectors. Offline, fast, no model files."""

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _features(self, text: str):
        for tok in tokenize(text):
            yield tok, 1.0
            padded = f"#{tok}#"
            for i in range(len(padded) - 2):
                yield "c:" + padded[i:i + 3], 0.4

    def _encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, t in enumerate(texts):
            for feat, w in self._features(t):
                h = zlib.crc32(feat.encode("utf-8"))
                out[r, h % self.dim] += w if (h >> 31) & 1 else -w
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)

    def embed_passages(self, texts, batch_size: int = 256):
        return self._encode(texts)

    def embed_query(self, text):
        return self._encode([text])[0]


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, device: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.name = model_name
        self.e5 = "e5" in model_name.lower()   # e5 models expect prefixes

    def embed_passages(self, texts, batch_size: int = 64):
        if self.e5:
            texts = ["passage: " + t for t in texts]
        return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                 show_progress_bar=len(texts) > 1000).astype(np.float32)

    def embed_query(self, text):
        q = ("query: " + text) if self.e5 else text
        return self.model.encode([q], normalize_embeddings=True)[0].astype(np.float32)


def get_embedder(name: str):
    if name.startswith("hashing"):
        dim = int(name.split("-")[1]) if "-" in name else 1024
        return HashingEmbedder(dim)
    try:
        return SentenceTransformerEmbedder(name)
    except Exception as e:
        log.warning("Could not load '%s' (%s). Falling back to hashing embedder.", name, e)
        return HashingEmbedder()
