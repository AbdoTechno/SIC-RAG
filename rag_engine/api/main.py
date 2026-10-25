import os
from pathlib import Path
import yaml
from typing import Dict, Any
from fastapi import FastAPI
from pipeline.ingestion_pipeline import IngestionPipeline

# Import modular routers
from api.routers import health_router, documents_router, chunks_router


def load_default_config() -> Dict[str, Any]:
    config_path = Path(__file__).resolve().parent.parent / "configs" / "pipeline_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env_provider = os.getenv("RAG_STORE_PROVIDER")
    env_dsn = os.getenv("RAG_DB_DSN")
    if env_provider:
        cfg["document_store"]["provider"] = env_provider
    if env_dsn:
        cfg["document_store"]["params"]["dsn"] = env_dsn
    return cfg


def create_app() -> FastAPI:
    app = FastAPI(
        title="Enterprise RAG Engine",
        version="2.0.0",
        description="Modular REST API for document ingestion, chunking, and storage inspection in a general-purpose Retrieval-Augmented Generation (RAG) system.",
    )

    # Initialize Singleton Pipeline and store in App State for Dependency Injection
    app.state.pipeline = IngestionPipeline(load_default_config())

    # Register Routers
    app.include_router(health_router.router)
    app.include_router(documents_router.router)
    app.include_router(chunks_router.router)

    return app


app = create_app()