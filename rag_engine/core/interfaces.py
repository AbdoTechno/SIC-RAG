from abc import ABC, abstractmethod
from typing import List, Dict, Any
from core.schemas import RawDocument, DocumentChunk


#Dependency Inversion
class BaseFileParser(ABC):
    @abstractmethod
    def parse(self, file_path: str, base_metadata: Dict[str, Any]) -> List[RawDocument]:
        pass

class BaseTextNormalizer(ABC):
    @abstractmethod
    def normalize(self, text: str) -> str:
        pass

class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, document: RawDocument) -> List[DocumentChunk]:
        pass

class BaseEmbedder(ABC):
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        pass

class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        pass