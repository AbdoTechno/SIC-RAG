from fastapi import APIRouter, Depends, HTTPException, status
from api.dependencies import get_rag_pipeline
from api.schemas import RAGGenerateRequest, CaseReviewRequest, CaseReviewResponse
from api.errors import APIError
from pipeline.rag_pipeline import EndToEndRAGPipeline

router = APIRouter(tags=["RAG Generation (Groq LLM)"])


@router.get("/rag/models", status_code=status.HTTP_200_OK)
def get_available_llm_models(
    rag_pipeline: EndToEndRAGPipeline = Depends(get_rag_pipeline),
):
    try:
        available = (
            rag_pipeline.generator.list_available_models()
            if hasattr(rag_pipeline.generator, "list_available_models")
            else [getattr(rag_pipeline.generator, "model_name", "unknown")]
        )
        return {
            "configured_model": getattr(rag_pipeline.generator, "model_name", "unknown"),
            "available_models": available,
        }
    except Exception as e:
        raise APIError(503, "GENERATION_UNAVAILABLE", "The generation provider is unavailable.", retryable=True) from e


@router.post("/rag/generate", status_code=status.HTTP_200_OK)
def generate_rag_answer(
    req: RAGGenerateRequest,
    rag_pipeline: EndToEndRAGPipeline = Depends(get_rag_pipeline),
):
    try:
        res = rag_pipeline.run(
            query=req.query,
            top_k=req.top_k,
            mode=req.mode,
            file_id_filter=req.file_id_filter,
            abstention_threshold_override=req.abstention_threshold_override,
        )
        return {
            "query": res.query,
            "abstained": res.abstained,
            "model_name": res.model_name,
            "answer": res.answer,
            "token_usage": res.token_usage,
            "sources_cited": res.sources_cited,
        }
    except ValueError as exc:
        message = str(exc)
        if "API key" in message or message == "GENERATION_UNAVAILABLE":
            raise APIError(503, "GENERATION_UNAVAILABLE", "The generation provider is unavailable.", retryable=True) from exc
        raise APIError(400, "GENERATION_FAILED", "The grounded answer could not be generated.") from exc
    except Exception as exc:
        raise APIError(503, "GENERATION_UNAVAILABLE", "The generation provider is unavailable.", retryable=True) from exc


@router.post("/rag/case-review", response_model=CaseReviewResponse, status_code=status.HTTP_200_OK)
def generate_case_review(
    req: CaseReviewRequest,
    rag_pipeline: EndToEndRAGPipeline = Depends(get_rag_pipeline),
):
    knowledge = rag_pipeline.retrieval_pipeline.knowledge_status()
    if knowledge["registered_document_count"] == 0:
        raise APIError(400, "KNOWLEDGE_BASE_EMPTY", "No reference documents are registered.")
    if not knowledge["retrieval_ready"]:
        raise APIError(400, "INDEX_NOT_READY", "Reference documents are not fully indexed.", retryable=True)
    try:
        return rag_pipeline.run_case_review(req.model_dump())
    except ValueError as exc:
        code = str(exc)
        if code == "INVALID_MODEL_RESPONSE":
            raise APIError(502, code, "The generation provider returned an invalid structured response.", retryable=True)
        if code == "GENERATION_UNAVAILABLE":
            raise APIError(503, code, "The generation provider is unavailable.", retryable=True)
        raise APIError(400, "CASE_REVIEW_FAILED", "The case review could not be completed.")
    except Exception as exc:
        raise APIError(503, "GENERATION_UNAVAILABLE", "The generation provider is unavailable.", retryable=True) from exc
