import re
from typing import Dict, Any, Optional, List
from core.schemas import GenerationResult
from core.interfaces import BaseGenerator
from pipeline.retrieval_pipeline import RetrievalPipeline
from core.registry import registry
import components.generators


class EndToEndRAGPipeline:
    def __init__(
        self,
        config: Dict[str, Any],
        retrieval_pipeline: RetrievalPipeline,
        generator: Optional[BaseGenerator] = None,
    ):
        self.config = config
        self.retrieval_pipeline = retrieval_pipeline

        if generator is not None:
            self.generator = generator
        else:
            gen_cfg = config.get("generation", {"name": "groq_generator", "params": {}})
            gen_cls = registry.get_generator(gen_cfg["name"])
            self.generator = gen_cls(**gen_cfg.get("params", {}))

    def run(
        self,
        query: str,
        top_k: int = 3,
        mode: str = "hybrid",
        file_id_filter: Optional[str] = None,
        abstention_threshold_override: Optional[float] = None,
    ) -> GenerationResult:
        retrieval_res = self.retrieval_pipeline.retrieve(
            query=query,
            top_k=top_k,
            mode=mode,
            file_id_filter=file_id_filter,
            abstention_threshold_override=abstention_threshold_override,
        )

        if retrieval_res.abstained:
            return GenerationResult(
                query=query,
                answer=f"Abstained from answering: {retrieval_res.abstention_reason}",
                sources_cited=[],
                abstained=True,
                model_name="none",
                token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )

        gen_res = self.generator.generate(
            query=query,
            context_chunks=retrieval_res.ranked_chunks,
        )
        answer_text = gen_res["answer"]

        cited_indices = {
            int(m) for m in re.findall(r"\[Source\s+(\d+)\]", answer_text, flags=re.IGNORECASE)
        }

        sources = self._build_sources(retrieval_res.ranked_chunks, cited_indices)

        return GenerationResult(
            query=query,
            answer=answer_text,
            sources_cited=sources,
            abstained=False,
            model_name=gen_res["model_name"],
            token_usage=gen_res.get("token_usage", {}),
        )

    @staticmethod
    def _build_sources(chunks: List[Any], cited_indices: set[int]) -> List[Dict[str, Any]]:
        sources = []
        for idx, chunk in enumerate(chunks, start=1):
            extra = chunk.metadata.extra or {}
            sources.append({
                "source_ref": f"[Source {idx}]",
                "cited_in_answer": idx in cited_indices,
                "chunk_id": chunk.chunk_id,
                "file_id": chunk.file_id,
                "section": chunk.metadata.page_or_section,
                "source_name": chunk.metadata.source_name,
                "rerank_score": chunk.rerank_score,
                "snippet": chunk.text[:300] + ("..." if len(chunk.text) > 300 else ""),
                "pdf_page": extra.get("pdf_page"),
                "organization": extra.get("organization"),
                "publication_date": extra.get("publication_date"),
                "source_url": chunk.metadata.source_link if chunk.metadata.source_link != "db://raw_files" else None,
            })
        return sources

    @staticmethod
    def _focused_case_query(question: str, clinical_context: Dict[str, Any], imaging_findings: Dict[str, Any]) -> str:
        terms = [question.strip()]
        if clinical_context.get("chief_complaint"):
            terms.append(str(clinical_context["chief_complaint"]))
        terms.extend(str(item.get("name", "")) for item in clinical_context.get("symptoms", [])[:10])
        terms.extend(str(item.get("label", "")) for item in imaging_findings.get("findings", [])[:10])
        chronic = clinical_context.get("chronic_conditions", {})
        if chronic.get("status") == "known":
            for item in chronic.get("items", [])[:10]:
                terms.append(item.get("name", "") if isinstance(item, dict) else str(item))
        return " | ".join(term for term in terms if term)

    def run_case_review(self, request: Dict[str, Any]) -> Dict[str, Any]:
        query = self._focused_case_query(
            request["question"], request["clinical_context"], request["imaging_findings"]
        )
        retrieval = self.retrieval_pipeline.retrieve(
            query=query,
            top_k=request.get("top_k", 3),
            mode=request.get("mode", "hybrid"),
            file_id_filter=request.get("file_id_filter"),
        )
        if retrieval.abstained:
            return {
                "request_id": request["request_id"], "assessment_id": request["assessment_id"],
                "input_revision": request["input_revision"], "question": request["question"],
                "abstained": True, "abstention_reason": retrieval.abstention_reason,
                "model_name": "none", "answer": "",
                "case_summary": "The supplied assessment was received, but no supporting reference evidence was retrieved.",
                "evidence_discussion": "", "missing_information": [], "review_points": [],
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "sources_cited": [], "service": {"name": "rag-engine", "version": "3.1.0"},
            }
        generator_method = getattr(self.generator, "generate_case_review", None)
        if not callable(generator_method):
            raise ValueError("GENERATION_UNAVAILABLE")
        generated = generator_method(
            question=request["question"], clinical_context=request["clinical_context"],
            imaging_findings=request["imaging_findings"], context_chunks=retrieval.ranked_chunks,
        )
        required_strings = ("case_summary", "evidence_discussion")
        if any(not isinstance(generated.get(key), str) for key in required_strings):
            raise ValueError("INVALID_MODEL_RESPONSE")
        if not isinstance(generated.get("missing_information"), list) or not isinstance(generated.get("review_points"), list):
            raise ValueError("INVALID_MODEL_RESPONSE")
        if any(not isinstance(item, str) for item in generated["missing_information"]):
            raise ValueError("INVALID_MODEL_RESPONSE")
        if any(not isinstance(item, str) for item in generated["review_points"]):
            raise ValueError("INVALID_MODEL_RESPONSE")
        cited = {int(value) for value in re.findall(r"\[Source\s+(\d+)\]", generated["evidence_discussion"], re.I)}
        valid = set(range(1, len(retrieval.ranked_chunks) + 1))
        if not cited or not cited.issubset(valid):
            raise ValueError("INVALID_MODEL_RESPONSE")
        sources = self._build_sources(retrieval.ranked_chunks, cited)
        return {
            "request_id": request["request_id"], "assessment_id": request["assessment_id"],
            "input_revision": request["input_revision"], "question": request["question"],
            "abstained": False, "abstention_reason": None,
            "model_name": generated.get("model_name", getattr(self.generator, "model_name", "unknown")),
            "answer": generated["evidence_discussion"],
            "case_summary": generated["case_summary"],
            "evidence_discussion": generated["evidence_discussion"],
            "missing_information": [str(item) for item in generated["missing_information"]],
            "review_points": [str(item) for item in generated["review_points"]],
            "token_usage": generated.get("token_usage", {}), "sources_cited": sources,
            "service": {"name": "rag-engine", "version": "3.1.0"},
        }
