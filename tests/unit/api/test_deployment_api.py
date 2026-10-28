from fastapi.testclient import TestClient

from api.main import create_app, load_default_config
from core.registry import registry
from pipeline.ingestion_pipeline import IngestionPipeline
from pipeline.retrieval_pipeline import RetrievalPipeline


def _deployment_client(monkeypatch, api_key: str = "") -> TestClient:
    monkeypatch.setenv("RAG_API_KEY", api_key)
    cfg = load_default_config()
    store = registry.get_document_store("sqlite_store")(dsn=":memory:")
    vector_store = registry.get_vector_store("in_memory_vector_store")()
    ingest = IngestionPipeline(config=cfg, store=store)
    retrieval = RetrievalPipeline(config=cfg, document_store=store, vector_store=vector_store)
    return TestClient(create_app(pipeline=ingest, retrieval_pipeline=retrieval))


def test_root_and_health_are_public_when_api_key_is_configured(monkeypatch):
    client = _deployment_client(monkeypatch, api_key="test-secret")

    assert client.get("/").status_code == 200
    assert client.get("/health").status_code == 200


def test_private_routes_require_configured_api_key(monkeypatch):
    client = _deployment_client(monkeypatch, api_key="test-secret")

    assert client.get("/documents").status_code == 401
    assert client.get("/documents", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/documents", headers={"X-API-Key": "test-secret"}).status_code == 200
