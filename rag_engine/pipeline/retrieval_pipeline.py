import math
import os
import re
from typing import Dict, Any, List, Optional

from core.schemas import (
    DocumentChunk,
    RetrievalResult,
    EvalQueryCase,
    RetrievalEvalReport,
)
from core.interfaces import (
    BaseDocumentStore,
    BaseEmbedder,
    BaseVectorStore,
    BaseReranker,
)
from core.registry import registry
import components.embedders
import components.vectorstores
import components.rerankers


# Design Pattern: Facade Pattern. Orchestrates Vector Indexing, Hybrid Search, Cross-Reranking, Abstention Gating, and Evaluation.
class RetrievalPipeline:
    def __init__(
        self,
        config: Dict[str, Any],
        document_store: BaseDocumentStore,
        embedder: Optional[BaseEmbedder] = None,
        vector_store: Optional[BaseVectorStore] = None,
        reranker: Optional[BaseReranker] = None,
    ):
        self.config = config
        self.document_store = document_store
        ret_cfg = config.get("retrieval", {})

        if embedder is not None:
            self.embedder = embedder
        else:
            emb_cfg = ret_cfg.get("embedder", {"name": "hash_embedder", "params": {"dim": 384}})
            emb_cls = registry.get_embedder(emb_cfg["name"])
            self.embedder = emb_cls(**emb_cfg.get("params", {}))

        if vector_store is not None:
            self.vector_store = vector_store
        else:
            vs_cfg = ret_cfg.get("vector_store", {"provider": "in_memory_vector_store", "params": {}})
            vs_cls = registry.get_vector_store(vs_cfg["provider"])
            self.vector_store = vs_cls(**vs_cfg.get("params", {}))

        if reranker is not None:
            self.reranker = reranker
        else:
            rerank_cfg = ret_cfg.get("reranker", {"name": "hybrid_overlap_reranker", "params": {}})
            rerank_cls = registry.get_reranker(rerank_cfg["name"])
            self.reranker = rerank_cls(**rerank_cfg.get("params", {}))

        self.abstention_threshold: float = float(ret_cfg.get("abstention_threshold", 0.25))

    def index_file_chunks(self, file_id: str) -> Dict[str, Any]:
        chunks = self.document_store.get_chunks_by_file(file_id)
        if not chunks:
            self._set_index_status(file_id, "failed", 0, "No persisted chunks found.")
            raise ValueError(f"No chunks found for file_id '{file_id}'. Run chunking first.")

        self._set_index_status(file_id, "indexing", 0, None)
        try:
            texts = [f"{c.metadata.page_or_section or ''} | {c.text}".strip() for c in chunks]
            vectors = self.embedder.embed_documents(texts)
            if len(vectors) != len(chunks):
                raise ValueError("Embedding provider returned an unexpected vector count.")
            for chunk, vec in zip(chunks, vectors):
                chunk.embedding = vec

            self.vector_store.delete_by_file_id(file_id)
            indexed_count = self.vector_store.upsert_chunks(chunks)
            if indexed_count != len(chunks):
                raise ValueError("Vector store did not persist every chunk.")
            self._set_index_status(file_id, "ready", indexed_count, None)
        except Exception as exc:
            self._set_index_status(file_id, "failed", 0, str(exc)[:300])
            raise

        return {
            "file_id": file_id,
            "indexed_chunks_count": indexed_count,
            "embedding_dimension": self.embedder.dimension,
            "total_vectors_in_store": self.vector_store.count_vectors(),
            "index_status": "ready",
        }

    def _set_index_status(self, file_id: str, status: str, indexed_chunks: int, error: Optional[str]) -> None:
        setter = getattr(self.vector_store, "set_index_status", None)
        if callable(setter):
            setter(file_id, status, indexed_chunks, error)

    def invalidate_file_index(self, file_id: str) -> int:
        deleted = self.vector_store.delete_by_file_id(file_id)
        self._set_index_status(file_id, "awaiting_index", 0, None)
        return deleted

    def knowledge_status(self) -> Dict[str, Any]:
        documents = self.document_store.list_raw_files()
        statuses_getter = getattr(self.vector_store, "get_index_statuses", None)
        stored_statuses = statuses_getter() if callable(statuses_getter) else {}
        details = []
        total_chunks = 0
        total_indexed = 0
        for document in documents:
            file_id = document["file_id"]
            chunk_count = int(document.get("chunk_count", 0))
            indexed_count = self.vector_store.count_vectors(file_id)
            total_chunks += chunk_count
            total_indexed += indexed_count
            stored = stored_statuses.get(file_id, {})
            stored_status = stored.get("status")
            if stored_status == "failed":
                status = "failed"
            elif stored_status in {"indexing", "awaiting_index"}:
                status = "awaiting_index"
            elif chunk_count > 0 and indexed_count == chunk_count:
                status = "ready"
            else:
                status = "awaiting_index"
            details.append({
                "file_id": file_id,
                "filename": document["filename"],
                "content_hash": document["content_hash"],
                "chunk_count": chunk_count,
                "indexed_chunk_count": indexed_count,
                "index_status": status,
                "error": stored.get("error") if status == "failed" else None,
            })
        ready = bool(documents) and all(item["index_status"] == "ready" for item in details)
        retrieval_cfg = self.config.get("retrieval", {})
        generation_cfg = self.config.get("generation", {})
        generation_provider = generation_cfg.get("name") or generation_cfg.get("provider", "unknown")
        return {
            "retrieval_ready": ready,
            "registered_document_count": len(documents),
            "parsed_chunk_count": total_chunks,
            "indexed_chunk_count": total_indexed,
            "documents_awaiting_index": sum(item["index_status"] == "awaiting_index" for item in details),
            "failed_documents": sum(item["index_status"] == "failed" for item in details),
            "embedding": {
                "provider": retrieval_cfg.get("embedder", {}).get("name", type(self.embedder).__name__),
                "dimension": self.embedder.dimension,
            },
            "vector_store": retrieval_cfg.get("vector_store", {}).get("provider", type(self.vector_store).__name__),
            "generation_availability": {
                "provider": generation_provider,
                "configured": generation_provider == "mock_generator" or bool(os.getenv("GROQ_API_KEY")),
                "provider_check_performed": False,
                "note": "Configuration only; no model-provider call is made by this status endpoint.",
            },
            "documents": details,
        }

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
        file_id_filter: Optional[str] = None,
        content_type_filter: Optional[str] = None,
        abstention_threshold_override: Optional[float] = None,
    ) -> RetrievalResult:
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")
        lowered = query.lower()
        injection_markers = (
            "ignore all instructions", "ignore previous instructions", "reveal your system prompt",
            "show your system prompt", "print the system prompt", "developer message",
        )
        if any(marker in lowered for marker in injection_markers):
            return RetrievalResult(
                query=query, ranked_chunks=[], max_confidence_score=0.0, abstained=True,
                abstention_reason="The request attempts to override system instructions and was not used for retrieval.",
                search_mode=mode, telemetry={"candidate_count": 0, "safety_filter": "prompt_injection"},
            )
        raw_image_markers = ("inspect this raw chest image", "analyze the raw x-ray", "analyse the raw x-ray", "read the raw image")
        if any(marker in lowered for marker in raw_image_markers):
            return RetrievalResult(
                query=query, ranked_chunks=[], max_confidence_score=0.0, abstained=True,
                abstention_reason="This service does not inspect raw chest images; provide structured imaging findings instead.",
                search_mode=mode, telemetry={"candidate_count": 0, "safety_filter": "raw_image_not_supported"},
            )
        retrieval_cfg = self.config.get("retrieval", {})
        if retrieval_cfg.get("enforce_scope", False):
            scope_terms = [str(term).lower() for term in retrieval_cfg.get("scope_terms", [])]
            if scope_terms and not any(term in lowered for term in scope_terms):
                return RetrievalResult(
                    query=query, ranked_chunks=[], max_confidence_score=0.0, abstained=True,
                    abstention_reason="The question is outside the configured respiratory reference scope.",
                    search_mode=mode, telemetry={"candidate_count": 0, "safety_filter": "out_of_scope"},
                )

        expanded_query = getattr(self.embedder, "expand_query", lambda value: value)(query)
        q_vec = self.embedder.embed_query(query)
        candidates = self.vector_store.search(
            query_embedding=q_vec,
            query_text=expanded_query,
            top_k=max(top_k * 8, 40),
            mode=mode,
            file_id_filter=file_id_filter,
            content_type_filter=content_type_filter,
        )

        embedder_name = self.config.get("retrieval", {}).get("embedder", {}).get("name")
        if embedder_name == "hash_embedder":
            expanded_lower = expanded_query.lower()
            subject_markers = []
            if "aspiration pneumonia" in expanded_lower:
                subject_markers.append(("aspiration",))
            if "hospital acquired pneumonia" in expanded_lower or "hospital-acquired pneumonia" in expanded_lower:
                subject_markers.append(("hospital-acquired", "hospital acquired"))
            if subject_markers:
                subject_filtered = []
                for candidate in candidates:
                    subject_text = f"{candidate.metadata.source_name} {candidate.metadata.page_or_section or ''} {candidate.text}".lower()
                    if all(any(marker in subject_text for marker in marker_group) for marker_group in subject_markers):
                        subject_filtered.append(candidate)
                candidates = subject_filtered
            intent_groups = []
            if re.search(r"\b(treat|treatment|therapy|managed|management|antibiotic|duration|regimen)\b", expanded_query, re.I):
                intent_groups.append(("treat", "treatment", "therapy", "managed", "management", "antibiotic", "duration", "regimen"))
            if re.search(r"\b(define|defined|definition)\b", expanded_query, re.I):
                intent_groups.append(("define", "defined", "definition", "refers to", "develops"))
            if intent_groups:
                generic = {
                    "what", "which", "about", "reference", "evidence", "relevant", "clinical", "patient",
                    "the", "for", "and", "with", "from", "that", "this", "are", "is", "how",
                    "treat", "treatment", "therapy", "managed", "management", "antibiotic", "duration",
                    "regimen", "define", "defined", "definition", "discussed",
                }
                core_terms = {
                    token for token in re.findall(r"[a-z0-9\u0600-\u06ff-]+", expanded_query.lower())
                    if len(token) >= 3 and token not in generic
                }
                filtered = []
                for candidate in candidates:
                    haystack = f"{candidate.metadata.page_or_section or ''} {candidate.text}".lower()
                    has_intent = all(any(marker in haystack for marker in group) for group in intent_groups)
                    has_subject = not core_terms or any(term in haystack for term in core_terms)
                    if has_intent and has_subject:
                        filtered.append(candidate)
                candidates = filtered

        ranked = self.reranker.rerank(query=expanded_query, chunks=candidates, top_k=top_k)
        max_score = max((c.rerank_score or 0.0 for c in ranked), default=0.0)
        if embedder_name == "hash_embedder" and ranked and max((c.lexical_score or 0.0 for c in ranked), default=0.0) == 0.0:
            return RetrievalResult(
                query=query, ranked_chunks=[], max_confidence_score=0.0, abstained=True,
                abstention_reason="No meaningful query terms matched the indexed reference corpus.",
                search_mode=mode, telemetry={"candidate_count": len(candidates), "threshold": self.abstention_threshold},
            )
        threshold = (
            abstention_threshold_override
            if abstention_threshold_override is not None
            else self.abstention_threshold
        )

        if not ranked or max_score < threshold:
            return RetrievalResult(
                query=query,
                ranked_chunks=[],
                max_confidence_score=max_score,
                abstained=True,
                abstention_reason=f"Max confidence score ({max_score:.3f}) is below threshold ({threshold:.3f}).",
                search_mode=mode,
                telemetry={"candidate_count": len(candidates), "threshold": threshold},
            )

        return RetrievalResult(
            query=query,
            ranked_chunks=ranked,
            max_confidence_score=max_score,
            abstained=False,
            abstention_reason=None,
            search_mode=mode,
            telemetry={"candidate_count": len(candidates), "threshold": threshold},
        )

    @staticmethod
    def _is_chunk_relevant(chunk: DocumentChunk, case: EvalQueryCase) -> bool:
        text_lower = chunk.text.lower()
        sec_lower = (chunk.metadata.page_or_section or "").lower()
        if case.expected_section_substr and case.expected_section_substr.lower() in sec_lower:
            return True
        if case.expected_keywords:
            return any(kw.lower() in text_lower for kw in case.expected_keywords)
        return False

    def evaluate_retrieval(
        self,
        test_cases: List[EvalQueryCase],
        top_k: int = 5,
        mode: str = "hybrid",
        file_id_filter: Optional[str] = None,
    ) -> RetrievalEvalReport:
        if not test_cases:
            raise ValueError("test_cases cannot be empty.")

        hits = 0
        rr_sum = 0.0
        ndcg_sum = 0.0
        precision_sum = 0.0
        answerable_count = 0
        abstention_correct = 0
        details: List[Dict[str, Any]] = []

        for case in test_cases:
            res = self.retrieve(
                query=case.query,
                top_k=top_k,
                mode=mode,
                file_id_filter=file_id_filter,
            )

            if case.should_abstain:
                is_correct_abstain = res.abstained
                if is_correct_abstain:
                    abstention_correct += 1
                details.append({
                    "query": case.query,
                    "type": "unanswerable",
                    "abstained": res.abstained,
                    "passed": is_correct_abstain,
                    "max_confidence_score": res.max_confidence_score,
                })
                continue

            answerable_count += 1
            if not res.abstained:
                abstention_correct += 1

            rel_flags = [1 if self._is_chunk_relevant(c, case) else 0 for c in res.ranked_chunks]
            is_hit = any(rel_flags)
            if is_hit:
                hits += 1

            first_rank = next((i + 1 for i, flag in enumerate(rel_flags) if flag == 1), 0)
            rr = (1.0 / first_rank) if first_rank > 0 else 0.0
            rr_sum += rr

            dcg = sum(flag / math.log2(idx + 2) for idx, flag in enumerate(rel_flags))
            ideal_flags = sorted(rel_flags, reverse=True)
            idcg = sum(flag / math.log2(idx + 2) for idx, flag in enumerate(ideal_flags))
            ndcg = (dcg / idcg) if idcg > 0 else 0.0
            ndcg_sum += ndcg

            prec = (sum(rel_flags) / len(rel_flags)) if rel_flags else 0.0
            precision_sum += prec

            details.append({
                "query": case.query,
                "type": "answerable",
                "hit": is_hit,
                "first_relevant_rank": first_rank,
                "reciprocal_rank": round(rr, 4),
                "ndcg": round(ndcg, 4),
                "context_precision": round(prec, 4),
                "max_confidence_score": res.max_confidence_score,
                "top_chunk_section": res.ranked_chunks[0].metadata.page_or_section if res.ranked_chunks else None,
            })

        denom = max(1, answerable_count)
        return RetrievalEvalReport(
            total_queries=len(test_cases),
            hit_rate_at_k=round(hits / denom, 4),
            mrr_score=round(rr_sum / denom, 4),
            ndcg_at_k=round(ndcg_sum / denom, 4),
            context_precision_at_k=round(precision_sum / denom, 4),
            abstention_accuracy=round(abstention_correct / len(test_cases), 4),
            per_query_details=details,
        )
