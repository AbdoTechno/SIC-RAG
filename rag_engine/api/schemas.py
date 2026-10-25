from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class ChunkingRequest(BaseModel):
    force_rechunk: bool = Field(default=False, description="Force re-chunking even if config matches")
    chunker_override: Optional[Dict[str, Any]] = Field(default=None, description="Override chunker settings")
    normalizers_override: Optional[List[Dict[str, Any]]] = Field(default=None, description="Override normalizer chain")


class ChunkPreview(BaseModel):
    chunk_id: str
    chunk_index: int
    char_count: int
    token_estimate: int
    content_type: str
    section: Optional[str]
    text: str


class ChunkingResponse(BaseModel):
    file_id: str
    status: str
    config_hash: str
    chunk_count: int
    chunks_preview: List[ChunkPreview]


class DocumentHealthResponse(BaseModel):
    status: str
    backend: str
    raw_files_count: int
    total_chunks_count: int


class ChunkListResponse(BaseModel):
    file_id: str
    filename: str
    total_chunks_in_db: int
    returned_count: int
    limit: int
    offset: int
    chunks: List[Dict[str, Any]]