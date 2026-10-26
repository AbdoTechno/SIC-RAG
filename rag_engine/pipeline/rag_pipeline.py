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

        sources: List[Dict[str, Any]] = []
        for idx, c in enumerate(retrieval_res.ranked_chunks, start=1):
            sources.append(
                {
                    "source_ref": f"[Source {idx}]",
                    "cited_in_answer": idx in cited_indices,
                    "chunk_id": c.chunk_id,
                    "file_id": c.file_id,
                    "section": c.metadata.page_or_section,
                    "source_name": c.metadata.source_name,
                    "rerank_score": c.rerank_score,
                    "snippet": c.text[:200] + ("..." if len(c.text) > 200 else ""),
                }
            )

        return GenerationResult(
            query=query,
            answer=answer_text,
            sources_cited=sources,
            abstained=False,
            model_name=gen_res["model_name"],
            token_usage=gen_res.get("token_usage", {}),
        )
