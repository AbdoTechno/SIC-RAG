import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import uuid

#immutability (Value Object) - prevents accidental mutation of source provenance across pipeline stages
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
            raise ValueError("source_name cannot be empty.")
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
    file_id: Optional[str] = None
    chunk_index: int = 0
    char_count: int = field(init=False)
    token_estimate: int = field(init=False)
    content_type: str = "prose"
    config_hash: str = "default"
    embedding: Optional[List[float]] = None
    rerank_score: Optional[float] = None

    def __post_init__(self):
        self.char_count = len(self.text)
        # Fast heuristic approximation (~4 chars per token for multilingual/English mix)
        self.token_estimate = max(1, len(self.text.split())) if self.text.strip() else 0