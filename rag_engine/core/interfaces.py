from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from core.schemas import StoredRawFile, RawDocument, DocumentChunk

class BaseFileParser(ABC):
    @abstractmethod
    def parse(self, raw_bytes: bytes, base_metadata: Dict[str, Any], file_id: Optional[str] = None) -> List[RawDocument]:
        pass

class BaseDocumentStore(ABC):
    @abstractmethod
    def save_raw_file(self, raw_file: StoredRawFile) -> str:
        pass

    @abstractmethod
    def get_raw_file(self, file_id: str) -> Optional[StoredRawFile]:
        pass

    @abstractmethod
    def save_parsed_documents(self, documents: List[RawDocument]) -> None:
        pass

    @abstractmethod
    def get_parsed_documents_by_file(self, file_id: str) -> List[RawDocument]:
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