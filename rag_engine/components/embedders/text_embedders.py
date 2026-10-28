import hashlib
import math
import re
from typing import List
from core.interfaces import BaseEmbedder
from core.registry import registry


@registry.register_embedder("hash_embedder")
class DeterministicHashEmbedder(BaseEmbedder):
    ARABIC_QUERY_EXPANSIONS = {
        "الالتهاب الرئوي المكتسب من المستشفى": "hospital acquired pneumonia",
        "التهاب رئوي مكتسب من المستشفى": "hospital acquired pneumonia",
        "الالتهاب الرئوي": "pneumonia",
        "الارتشاح البلوري": "pleural effusion",
        "الانصباب الجنبي": "pleural effusion",
        "استرواح الصدر": "pneumothorax",
        "الالتهاب الرئوي بالشفط": "aspiration pneumonia",
        "تعريف": "definition defined",
    }
    def __init__(self, dim: int = 384):
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def _embed_single(self, text: str) -> List[float]:
        vec = [0.0] * self._dim
        tokens = re.findall(r"[a-zA-Z0-9_\u0600-\u06FF]{2,}", text.lower())
        if not tokens:
            return vec
        for token in tokens:
            # Simple stem normalization so "transformers" matches "transformer"
            norm_tok = token.rstrip("s") if len(token) > 4 else token
            h = hashlib.sha256(norm_tok.encode("utf-8")).digest()
            idx = int.from_bytes(h[:4], "little") % self._dim
            sign = 1.0 if (h[4] % 2 == 0) else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [round(v / norm, 6) for v in vec]
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_single(t) for t in texts]

    def expand_query(self, query: str) -> str:
        expanded = query
        for arabic, english in self.ARABIC_QUERY_EXPANSIONS.items():
            if arabic in query:
                expanded += f" {english}"
        if "aspiration pneumonia" in expanded.lower():
            expanded += " AP"
        return expanded

    def embed_query(self, query: str) -> List[float]:
        return self._embed_single(self.expand_query(query))


@registry.register_embedder("fastembed_embedder")
class FastEmbedONNXEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", dim: int = 384):
        from fastembed import TextEmbedding
        self._model_name = model_name
        self._dim = dim
        self._model = TextEmbedding(model_name=model_name)

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = list(self._model.embed(texts))
        return [[float(x) for x in vec[: self._dim]] for vec in embeddings]

    def embed_query(self, query: str) -> List[float]:
        embeddings = list(self._model.query_embed(query))
        return [float(x) for x in embeddings[0][: self._dim]]
