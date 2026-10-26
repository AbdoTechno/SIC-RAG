import re
from dataclasses import replace
from typing import List, Optional
from core.interfaces import BaseReranker
from core.schemas import DocumentChunk
from core.registry import registry


@registry.register_reranker("hybrid_overlap_reranker")
class HybridOverlapReranker(BaseReranker):
    def __init__(self, semantic_weight: float = 0.45, lexical_weight: float = 0.55):
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight

    def _compute_rerank_score(self, query: str, chunk: DocumentChunk) -> float:
        q_tokens = re.findall(r"[a-z0-9\u0600-\u06ff]+(?:-[a-z0-9\u0600-\u06ff]+)*", query.lower())
        if not q_tokens:
            return 0.0

        text_lower = chunk.text.lower()
        section_lower = (chunk.metadata.page_or_section or "").lower()
        combined_text = f"{section_lower} {text_lower}"

        matched = sum(1 for t in set(q_tokens) if t in combined_text)
        coverage = matched / len(set(q_tokens))

        # Bigram phrase bonus for high precision
        bigrams = [f"{q_tokens[i]} {q_tokens[i+1]}" for i in range(len(q_tokens) - 1)]
        phrase_bonus = 0.15 if (bigrams and any(bg in combined_text for bg in bigrams)) else 0.0

        sem_sim = max(0.0, chunk.similarity_score or 0.0)
        raw_score = (self.lexical_weight * coverage) + (self.semantic_weight * sem_sim) + phrase_bonus
        return round(min(1.0, raw_score), 4)

    def rerank(self, query: str, chunks: List[DocumentChunk], top_k: Optional[int] = None) -> List[DocumentChunk]:
        if not chunks:
            return []

        reranked = [
            replace(c, rerank_score=self._compute_rerank_score(query, c))
            for c in chunks
        ]
        reranked.sort(key=lambda x: x.rerank_score or 0.0, reverse=True)
        return reranked[:top_k] if top_k else reranked
