import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import uuid

@dataclass(frozen=True)
class SourceMetadata:
    source_name: str
    version_date: str
    source_link: str
    file_format: str
    page_or_section: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.source_name or not self.source_name.strip():
            raise ValueError("source_name cannot be empty for clinical evidence tracking.")
        if not self.file_format or not self.file_format.strip():
            raise ValueError("file_format must be specified.")

@dataclass(frozen=True)
class StoredRawFile:
    filename: str
    file_format: str
    raw_bytes: bytes
    metadata: SourceMetadata
    file_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content_hash: str = field(init=False)

    def __post_init__(self):
        if not self.raw_bytes:
            raise ValueError("raw_bytes cannot be empty.")
        computed_hash = hashlib.sha256(self.raw_bytes).hexdigest()
        object.__setattr__(self, "content_hash", computed_hash)

@dataclass
class RawDocument:
    content: str
    metadata: SourceMetadata
    file_id: Optional[str] = None
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        if not isinstance(self.metadata, SourceMetadata):
            raise TypeError("metadata must be an instance of SourceMetadata.")

@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: SourceMetadata
    embedding: Optional[List[float]] = None
    rerank_score: Optional[float] = None