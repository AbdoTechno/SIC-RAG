import hashlib
import math
import re
from typing import List
from core.interfaces import BaseEmbedder
from core.registry import registry


# System Design Decision: Feature Hashing (Weinberger et al.) projects arbitrary tokens & n-grams into a fixed L2-normalized vector space with zero external model weights.
@registry.register_embedder("hash_embedder")
class DeterministicHashEmbedder(BaseEmbedder):
    def __init__(self, dim: int = 384):
        if dim <= 0:
            raise ValueError("Embedding dimension must be strictly positive.")
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def _tokenize(self, text: str) -> List[str]:
        cleaned = text.lower()
        words = re.findall(r"[a-z0-9\u0600-\u06ff]+(?:-[a-z0-9\u0600-\u06ff]+)*", cleaned)
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
        return words + bigrams

    def _encode_single(self, text: str) -> List[float]:
        vec = [0.0] * self._dim
        tokens = self._tokenize(text)
        if not tokens:
            return vec

        for token in tokens:
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self._dim
            sign = 1.0 if ((h >> 16) & 1) == 0 else -1.0
            vec[idx] += sign

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [round(v / norm, 6) for v in vec]
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._encode_single(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        return self._encode_single(query)


@registry.register_embedder("fastembed_embedder")
class FastEmbedONNXEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", dim: int = 384):
        self.model_name = model_name
        self._dim = dim
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=self.model_name)

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self._ensure_model()
        return [list(map(float, vec)) for vec in self._model.embed(texts)]

    def embed_query(self, query: str) -> List[float]:
        self._ensure_model()
        return list(map(float, next(self._model.embed([query]))))
