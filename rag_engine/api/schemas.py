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


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query")
    top_k: int = Field(default=5, ge=1, le=50)
    mode: str = Field(default="hybrid", pattern="^(hybrid|semantic|lexical)$")
    file_id_filter: Optional[str] = None
    content_type_filter: Optional[str] = None
    abstention_threshold_override: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class EvalCaseInput(BaseModel):
    query: str
    expected_keywords: List[str] = Field(default_factory=list)
    expected_section_substr: Optional[str] = None
    should_abstain: bool = False


class RetrievalEvaluationRequest(BaseModel):
    top_k: int = Field(default=5, ge=1, le=20)
    mode: str = Field(default="hybrid", pattern="^(hybrid|semantic|lexical)$")
    file_id_filter: Optional[str] = None
    test_cases: List[EvalCaseInput]
