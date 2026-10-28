from fastapi.testclient import TestClient

from api.main import create_app, load_default_config
from core.registry import registry
from pipeline.ingestion_pipeline import IngestionPipeline
from pipeline.retrieval_pipeline import RetrievalPipeline


def test_chunk_changes_invalidate_existing_index(monkeypatch):
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    cfg = load_default_config()
    store = registry.get_document_store("sqlite_store")(dsn=":memory:")
    vector = registry.get_vector_store("in_memory_vector_store")()
    ingest = IngestionPipeline(cfg, store=store)
    retrieval = RetrievalPipeline(cfg, store, vector_store=vector)

    with TestClient(create_app(ingest, retrieval)) as client:
        upload = client.post(
            "/documents",
            files={"file": ("reference.txt", b"Pleural effusion clinical reference text.", "text/plain")},
            data={"source_name": "Reference", "auto_chunk": "true"},
        )
        assert upload.status_code == 201
        file_id = upload.json()["file_id"]
        status_before_index = client.get("/knowledge/status").json()
        assert status_before_index["documents"][0]["index_status"] == "awaiting_index"

        indexed = client.post(f"/documents/{file_id}/index")
        assert indexed.status_code == 200
        assert client.get("/knowledge/status").json()["retrieval_ready"] is True

        rechunk = client.post(
            f"/documents/{file_id}/chunk",
            json={
                "force_rechunk": True,
                "chunker_override": {
                    "name": "recursive_chunker",
                    "params": {"chunk_size": 30, "chunk_overlap": 5},
                },
            },
        )
        assert rechunk.status_code == 200
        readiness = client.get("/knowledge/status").json()
        assert readiness["retrieval_ready"] is False
        assert readiness["documents"][0]["index_status"] == "awaiting_index"
        assert readiness["documents"][0]["indexed_chunk_count"] == 0
