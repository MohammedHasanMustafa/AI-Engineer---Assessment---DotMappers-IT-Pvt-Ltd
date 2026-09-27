from __future__ import annotations

import hashlib

import numpy as np

from .text_utils import tokenize


class SentenceTransformerEmbedder:
    """Bi-encoder. BGE models expect an instruction prefix on queries only."""

    def __init__(self, model_name: str, query_prefix: str | None = None):
        self.model_name = model_name
        self._model = None
        if query_prefix is None:
            query_prefix = ("Represent this sentence for searching relevant passages: "
                            if "bge" in model_name.lower() else "")
        self.query_prefix = query_prefix

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        return self.model.encode(texts, batch_size=32, normalize_embeddings=True,
                                 show_progress_bar=False, convert_to_numpy=True).astype("float32")

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Symmetric sentence embeddings (grounding checks, extractive answers)."""
        return self._encode(texts)

    def embed_query(self, query: str) -> np.ndarray:
        return self._encode([self.query_prefix + query])[0]


class CrossEncoderReranker:
    def __init__(self, model_name: str, max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, max_length=self.max_length)
        return self._model

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0, dtype="float32")
        return np.asarray(self.model.predict([(query, t) for t in texts], show_progress_bar=False),
                          dtype="float32")


class HashingEmbedder:
    """Deterministic bag-of-words embedder. Used by unit tests (no model download)."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.model_name = "hashing-test-embedder"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        toks = tokenize(text)
        for t in toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]:
            v[int(hashlib.md5(t.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts):
        return np.vstack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), "float32")

    embed_texts = embed_documents

    def embed_query(self, query):
        return self._vec(query)
