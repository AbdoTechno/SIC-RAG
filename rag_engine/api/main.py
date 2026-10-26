import os
from pathlib import Path
import yaml
from typing import Dict, Any, Optional
from fastapi import FastAPI
from pipeline.ingestion_pipeline import IngestionPipeline
from pipeline.retrieval_pipeline import RetrievalPipeline

from api.routers import health_router, documents_router, chunks_router, retrieval_router


def load_default_config() -> Dict[str, Any]:
    config_path = Path(__file__).resolve().parent.parent / "configs" / "pipeline_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env_provider = os.getenv("RAG_STORE_PROVIDER")
    env_dsn = os.getenv("RAG_DB_DSN")
    if env_provider:
        cfg["document_store"]["provider"] = env_provider
        if env_provider == "sqlite_store":
            cfg.setdefault("retrieval", {})["vector_store"] = {
                "provider": "in_memory_vector_store",
                "params": {},
            }
    if env_dsn:
        cfg["document_store"]["params"]["dsn"] = env_dsn
        if cfg.get("retrieval", {}).get("vector_store", {}).get("provider") == "pgvector_store":
            cfg["retrieval"]["vector_store"]["params"]["dsn"] = env_dsn
    return cfg


def create_app(
    pipeline: Optional[IngestionPipeline] = None,
    retrieval_pipeline: Optional[RetrievalPipeline] = None,
) -> FastAPI:
    app = FastAPI(
        title="Enterprise RAG Engine - Ingestion, Retrieval & Evaluation API",
        version="2.5.0",
        description="Modular REST API for document ingestion, idempotent chunking, hybrid retrieval, and RAG evaluation.",
    )

    cfg = load_default_config()
    active_ingest = pipeline or IngestionPipeline(cfg)
    active_retrieval = retrieval_pipeline or RetrievalPipeline(
        config=cfg,
        document_store=active_ingest.store,
    )

    app.state.pipeline = active_ingest
    app.state.retrieval_pipeline = active_retrieval

    app.include_router(health_router.router)
    app.include_router(documents_router.router)
    app.include_router(chunks_router.router)
    app.include_router(retrieval_router.router)

    return app


app = create_app()
