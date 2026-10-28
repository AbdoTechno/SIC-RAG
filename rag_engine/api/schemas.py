from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Dict, Any, Optional, Literal, Union


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


def _clean_str_opt(v: Optional[str]) -> Optional[str]:
    if v is None or str(v).strip() in ("", "string", "null", "None"):
        return None
    return v


def _clean_float_opt(v: Any) -> Optional[float]:
    if v is None or v == "" or float(v) >= 1.0:
        return None
    return float(v)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query")
    top_k: int = Field(default=5, ge=1, le=50)
    mode: str = Field(default="hybrid", pattern="^(hybrid|semantic|lexical)$")
    file_id_filter: Optional[str] = Field(default=None, json_schema_extra={"example": None})
    content_type_filter: Optional[str] = Field(default=None, json_schema_extra={"example": None})
    abstention_threshold_override: Optional[float] = Field(default=None, ge=0.0, le=1.0, json_schema_extra={"example": None})

    _val_str = field_validator("file_id_filter", "content_type_filter", mode="before")(_clean_str_opt)
    _val_flt = field_validator("abstention_threshold_override", mode="before")(_clean_float_opt)


class EvalCaseInput(BaseModel):
    query: str
    expected_keywords: List[str] = Field(default_factory=list)
    expected_section_substr: Optional[str] = None
    should_abstain: bool = False


class RetrievalEvaluationRequest(BaseModel):
    top_k: int = Field(default=5, ge=1, le=20)
    mode: str = Field(default="hybrid", pattern="^(hybrid|semantic|lexical)$")
    file_id_filter: Optional[str] = Field(default=None, json_schema_extra={"example": None})
    test_cases: List[EvalCaseInput]


class RAGGenerateRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question to answer using retrieved evidence")
    top_k: int = Field(default=3, ge=1, le=15, description="Number of top reranked chunks to pass to the LLM")
    mode: str = Field(default="hybrid", pattern="^(hybrid|semantic|lexical)$")
    file_id_filter: Optional[str] = Field(default=None, json_schema_extra={"example": None})
    abstention_threshold_override: Optional[float] = Field(default=None, ge=0.0, le=1.0, json_schema_extra={"example": None})

    _val_str = field_validator("file_id_filter", mode="before")(_clean_str_opt)
    _val_flt = field_validator("abstention_threshold_override", mode="before")(_clean_float_opt)


class NamedItem(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class SymptomItem(NamedItem):
    duration: Optional[str] = Field(default=None, max_length=80)


class KnownItems(BaseModel):
    status: Literal["unknown", "none", "known"] = "unknown"
    items: List[Union[str, NamedItem]] = Field(default_factory=list, max_length=50)

    @model_validator(mode="after")
    def validate_status_and_items(self):
        if self.status in {"unknown", "none"} and self.items:
            raise ValueError("items must be empty when status is unknown or none")
        return self


class VitalItem(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    value: Union[float, str]
    unit: Optional[str] = Field(default=None, max_length=32)
    recorded_at: Optional[str] = Field(default=None, max_length=64)


class OtherTestItem(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    result: str = Field(min_length=1, max_length=500)
    unit: Optional[str] = Field(default=None, max_length=32)


class ClinicalContext(BaseModel):
    age_years: Optional[float] = Field(default=None, ge=0, le=130, allow_inf_nan=False)
    sex: Optional[Literal["female", "male", "intersex", "unknown", "not_recorded"]] = None
    chief_complaint: Optional[str] = Field(default=None, max_length=500)
    symptoms: List[SymptomItem] = Field(default_factory=list, max_length=50)
    chronic_conditions: KnownItems = Field(default_factory=KnownItems)
    allergies: KnownItems = Field(default_factory=KnownItems)
    medications: KnownItems = Field(default_factory=KnownItems)
    vitals: List[VitalItem] = Field(default_factory=list, max_length=50)
    other_tests: List[OtherTestItem] = Field(default_factory=list, max_length=50)
    doctor_notes: str = Field(default="", max_length=4000)


class ImagingFinding(BaseModel):
    label: str = Field(min_length=1, max_length=160)
    score: Optional[float] = Field(default=None, allow_inf_nan=False)
    score_kind: Optional[Literal["probability", "confidence", "logit", "unknown"]] = None
    calibrated: bool = False

    @model_validator(mode="after")
    def validate_score_semantics(self):
        if self.score_kind in {"probability", "confidence"} and self.score is not None:
            if not 0.0 <= self.score <= 1.0:
                raise ValueError("probability and confidence scores must be between 0 and 1")
        return self


class ImagingFindings(BaseModel):
    origin: Literal["local_model", "doctor_recorded", "none"] = "local_model"
    model_name: Optional[str] = Field(default=None, max_length=120)
    model_version: Optional[str] = Field(default=None, max_length=80)
    findings: List[ImagingFinding] = Field(default_factory=list, max_length=50)
    warnings: List[str] = Field(default_factory=list, max_length=20)


class CaseReviewRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")
    assessment_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")
    input_revision: int = Field(ge=1, le=2_147_483_647)
    question: str = Field(min_length=1, max_length=2000)
    clinical_context: ClinicalContext
    imaging_findings: ImagingFindings
    top_k: int = Field(default=3, ge=1, le=10)
    mode: Literal["hybrid", "semantic", "lexical"] = "hybrid"
    file_id_filter: Optional[str] = Field(default=None, max_length=128)

    _clean_file = field_validator("file_id_filter", mode="before")(_clean_str_opt)


class CaseReviewSource(BaseModel):
    source_ref: str
    cited_in_answer: bool
    chunk_id: str
    file_id: str
    source_name: str
    section: Optional[str] = None
    pdf_page: Optional[int] = None
    snippet: str
    rerank_score: Optional[float] = None
    organization: Optional[str] = None
    publication_date: Optional[str] = None
    source_url: Optional[str] = None


class CaseReviewResponse(BaseModel):
    request_id: str
    assessment_id: str
    input_revision: int
    question: str
    abstained: bool
    abstention_reason: Optional[str]
    model_name: str
    answer: str
    case_summary: str
    evidence_discussion: str
    missing_information: List[str]
    review_points: List[str]
    token_usage: Dict[str, int]
    sources_cited: List[CaseReviewSource]
    service: Dict[str, str]
