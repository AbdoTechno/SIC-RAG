"""Abstract base classes defining uniform component interfaces."""

from abc import ABC, abstractmethod
from typing import List
from rag_engine.core.schemas import Document, Chunk, Evidence

class BaseParser(ABC):
    """Abstract interface for file parsers."""
    @abstractmethod
    def parse(self, file_path: str) -> Document:
        pass

class BaseChunker(ABC):
    """Abstract interface for text chunking strategies."""
    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        pass

class BaseEmbedder(ABC):
    """Abstract interface for embedding models."""
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        pass
