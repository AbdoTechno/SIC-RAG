from fastapi import Request
from pipeline.ingestion_pipeline import IngestionPipeline


def get_pipeline(request: Request) -> IngestionPipeline:
    """
    Dependency Injection for the RAG Ingestion Pipeline.
    Retrieves the instantiated pipeline from the FastAPI app state.
    """
    return request.app.state.pipeline