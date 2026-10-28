from fastapi.testclient import TestClient

from api.main import create_app, load_default_config
from core.registry import registry
from pipeline.ingestion_pipeline import IngestionPipeline
from pipeline.retrieval_pipeline import RetrievalPipeline
from pipeline.rag_pipeline import EndToEndRAGPipeline


def _app_with_reference(generator=None):
    cfg = load_default_config()
    store = registry.get_document_store("sqlite_store")(dsn=":memory:")
    vector = registry.get_vector_store("in_memory_vector_store")()
    embedder = registry.get_embedder("hash_embedder")(dim=384)
    ingest = IngestionPipeline(cfg, store=store)
    retrieval = RetrievalPipeline(cfg, store, embedder=embedder, vector_store=vector)
    generator = generator or registry.get_generator("extractive_mock_generator")()
    rag = EndToEndRAGPipeline(cfg, retrieval, generator=generator)
    uploaded = ingest.upload_raw_document(
        "reference.txt",
        b"Pleural effusion evaluation should be interpreted with the clinical context.",
        "BTS Pleural Guideline",
        "2023",
        "https://example.test/bts",
        extra_metadata={"pdf_page": 2, "organization": "BTS", "publication_date": "2023"},
    )
    ingest.run_chunking_for_file(uploaded["file_id"])
    retrieval.index_file_chunks(uploaded["file_id"])
    return create_app(ingest, retrieval, rag)


def _payload():
    return {
        "request_id": "mobile-request-1",
        "assessment_id": "assessment-opaque-1",
        "input_revision": 2,
        "question": "Which reference evidence is relevant to pleural effusion?",
        "clinical_context": {
            "age_years": 58,
            "sex": "male",
            "chief_complaint": "Shortness of breath",
            "symptoms": [{"name": "dyspnea", "duration": "3 days"}],
            "chronic_conditions": {"status": "unknown", "items": []},
            "allergies": {"status": "none", "items": []},
            "medications": {"status": "unknown", "items": []},
            "vitals": [], "other_tests": [], "doctor_notes": "",
        },
        "imaging_findings": {
            "origin": "local_model", "model_name": "local-cxr", "model_version": "1",
            "findings": [{"label": "pleural effusion", "score": 0.82, "score_kind": "confidence", "calibrated": False}],
            "warnings": [],
        },
        "top_k": 3, "mode": "hybrid", "file_id_filter": None,
    }


def test_case_review_returns_typed_correlation_and_real_source_provenance(monkeypatch):
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    with TestClient(_app_with_reference()) as client:
        response = client.post("/rag/case-review", json=_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "mobile-request-1"
    assert body["assessment_id"] == "assessment-opaque-1"
    assert body["input_revision"] == 2
    assert body["abstained"] is False
    assert body["sources_cited"][0]["pdf_page"] == 2
    assert body["sources_cited"][0]["source_name"] == "BTS Pleural Guideline"
    assert "[Source 1]" in body["evidence_discussion"]


def test_case_review_validation_error_is_machine_readable(monkeypatch):
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    payload = _payload()
    payload["clinical_context"]["age_years"] = 900
    with TestClient(_app_with_reference()) as client:
        response = client.post("/rag/case-review", json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    assert response.json()["error"]["request_id"]


def test_case_review_rejects_invalid_structured_model_output(monkeypatch):
    monkeypatch.delenv("RAG_API_KEY", raising=False)

    class InvalidGenerator:
        model_name = "broken-test-generator"

        def generate_case_review(self, **_):
            return {"case_summary": 42, "evidence_discussion": "[Source 99]"}

    with TestClient(_app_with_reference(generator=InvalidGenerator())) as client:
        response = client.post("/rag/case-review", json=_payload())
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "INVALID_MODEL_RESPONSE"


def test_case_review_reports_empty_knowledge_base(monkeypatch):
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    cfg = load_default_config()
    store = registry.get_document_store("sqlite_store")(dsn=":memory:")
    retrieval = RetrievalPipeline(cfg, store, vector_store=registry.get_vector_store("in_memory_vector_store")())
    ingest = IngestionPipeline(cfg, store=store)
    generator = registry.get_generator("extractive_mock_generator")()
    rag = EndToEndRAGPipeline(cfg, retrieval, generator=generator)
    with TestClient(create_app(ingest, retrieval, rag)) as client:
        response = client.post("/rag/case-review", json=_payload())
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "KNOWLEDGE_BASE_EMPTY"
