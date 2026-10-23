"""Ingestion pipeline handling parsing, normalization, chunking, and vector storage."""

from rag_engine.core.schemas import Document

class IngestionPipeline:
    """Orchestrates the document ingestion workflow."""
    
    def __init__(self, config: dict):
        self.config = config

    def run(self, file_path: str) -> None:
        """Execute the full ingestion process for a source file."""
        # TODO: Implement parsing, cleaning, chunking, embedding, and storage
        pass
