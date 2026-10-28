from fastapi import APIRouter, Depends

from api.dependencies import get_retrieval_pipeline
from pipeline.retrieval_pipeline import RetrievalPipeline


router = APIRouter(prefix="/knowledge", tags=["Knowledge Base"])


@router.get("/status")
def knowledge_status(
    retrieval_pipeline: RetrievalPipeline = Depends(get_retrieval_pipeline),
):
    return retrieval_pipeline.knowledge_status()
