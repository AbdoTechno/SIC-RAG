from core.registry import registry
from pipeline.ingestion_pipeline import IngestionPipeline
from pipeline.retrieval_pipeline import RetrievalPipeline


def _config(db_path: str):
    return {
        "document_store": {"provider": "sqlite_store", "params": {"dsn": db_path}},
        "ingestion": {
            "normalizers": [{"name": "unicode_whitespace_cleaner", "params": {}}],
            "chunker": {"name": "recursive_chunker", "params": {"chunk_size": 240, "chunk_overlap": 30}},
        },
        "retrieval": {
            "abstention_threshold": 0.1,
            "embedder": {"name": "hash_embedder", "params": {"dim": 384}},
            "vector_store": {"provider": "sqlite_vector_store", "params": {"dsn": db_path, "dimension": 384}},
            "reranker": {"name": "hybrid_overlap_reranker", "params": {}},
        },
        "generation": {"name": "extractive_mock_generator", "params": {}},
    }


def test_sqlite_index_survives_restart_and_duplicate_ingestion_is_idempotent(tmp_path):
    database_file = tmp_path / "missing" / "nested" / "rag.db"
    db_path = str(database_file)
    cfg = _config(db_path)
    ingest = IngestionPipeline(cfg)
    retrieval = RetrievalPipeline(cfg, ingest.store)
    assert database_file.is_file()
    raw = b"Pleural infection assessment includes pleural fluid pH and clinical review."
    first = ingest.upload_raw_document(
        "pleural.md", raw, "Pleural Guideline", "2023", "https://example.test/pleural",
        extra_metadata={"pdf_page": 12, "organization": "BTS"},
    )
    chunked = ingest.run_chunking_for_file(first["file_id"])
    indexed = retrieval.index_file_chunks(first["file_id"])
    assert indexed["indexed_chunks_count"] == chunked["chunk_count"]
    retrieval.vector_store.close()
    ingest.store.conn.close()

    ingest_after_restart = IngestionPipeline(cfg)
    retrieval_after_restart = RetrievalPipeline(cfg, ingest_after_restart.store)
    result = retrieval_after_restart.retrieve("pleural fluid pH", top_k=3)
    assert result.abstained is False
    assert result.ranked_chunks[0].file_id == first["file_id"]
    status = retrieval_after_restart.knowledge_status()
    assert status["retrieval_ready"] is True
    assert status["parsed_chunk_count"] == status["indexed_chunk_count"]

    duplicate = ingest_after_restart.upload_raw_document(
        "pleural-copy.md", raw, "Pleural Guideline", "2023", "https://example.test/pleural",
        extra_metadata={"review_state": "verified"},
    )
    cached = ingest_after_restart.run_chunking_for_file(duplicate["file_id"])
    assert duplicate["deduplicated"] is True
    assert duplicate["file_id"] == first["file_id"]
    assert ingest_after_restart.store.get_raw_file(first["file_id"]).metadata.extra["review_state"] == "verified"
    assert cached["status"] == "cached_skip"
    assert retrieval_after_restart.vector_store.count_vectors(first["file_id"]) == chunked["chunk_count"]


def test_failed_index_can_be_retried_without_reupload(tmp_path):
    db_path = str(tmp_path / "retry.db")
    cfg = _config(db_path)
    ingest = IngestionPipeline(cfg)
    uploaded = ingest.upload_raw_document("retry.txt", b"pneumonia evidence", "Reference")
    ingest.run_chunking_for_file(uploaded["file_id"])

    class BrokenEmbedder:
        dimension = 384
        def embed_documents(self, texts):
            return []
        def embed_query(self, query):
            return [0.0] * 384

    broken = RetrievalPipeline(cfg, ingest.store, embedder=BrokenEmbedder())
    try:
        broken.index_file_chunks(uploaded["file_id"])
        assert False, "indexing should fail"
    except ValueError:
        pass
    assert broken.knowledge_status()["failed_documents"] == 1

    working = RetrievalPipeline(cfg, ingest.store)
    assert working.index_file_chunks(uploaded["file_id"])["index_status"] == "ready"
    assert working.knowledge_status()["retrieval_ready"] is True
