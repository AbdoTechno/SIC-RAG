import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
import yaml
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pipeline.ingestion_pipeline import IngestionPipeline
from pipeline.retrieval_pipeline import RetrievalPipeline
from pipeline.rag_pipeline import EndToEndRAGPipeline

from api.routers import (
    health_router,
    documents_router,
    chunks_router,
    retrieval_router,
    generation_router,
)

load_dotenv()


def load_default_config() -> Dict[str, Any]:
    config_path = Path(__file__).resolve().parent.parent / "configs" / "pipeline_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Database Overrides
    env_provider = os.getenv("RAG_STORE_PROVIDER")
    env_dsn = os.getenv("RAG_DB_DSN")
    if env_provider:
        cfg["document_store"]["provider"] = env_provider
        if env_provider == "sqlite_store":
            cfg["document_store"]["params"] = {"dsn": env_dsn or ":memory:"}
            cfg.setdefault("retrieval", {})["vector_store"] = {
                "provider": "in_memory_vector_store",
                "params": {},
            }
    if env_dsn and env_provider != "sqlite_store":
        cfg["document_store"]["params"]["dsn"] = env_dsn
        if cfg.get("retrieval", {}).get("vector_store", {}).get("provider") == "pgvector_store":
            cfg["retrieval"]["vector_store"]["params"]["dsn"] = env_dsn

    env_pool_max = os.getenv("RAG_DB_POOL_MAX")
    if env_pool_max and cfg["document_store"]["provider"] == "postgres_store":
        pool_max = int(env_pool_max)
        if pool_max < 1:
            raise ValueError("RAG_DB_POOL_MAX must be at least 1.")
        cfg["document_store"]["params"]["max_size"] = pool_max
        vector_params = cfg.get("retrieval", {}).get("vector_store", {}).get("params", {})
        if vector_params:
            vector_params["max_size"] = pool_max
            
    # SWE Fix: LLM Generation Overrides via Environment Variables
    env_llm_model = os.getenv("RAG_LLM_MODEL")
    env_llm_temp = os.getenv("RAG_LLM_TEMP")
    env_llm_tokens = os.getenv("RAG_LLM_MAX_TOKENS")
    
    if env_llm_model or env_llm_temp or env_llm_tokens:
        gen_params = cfg.setdefault("generation", {}).setdefault("params", {})
        if env_llm_model:
            gen_params["model_name"] = env_llm_model
        if env_llm_temp:
            gen_params["temperature"] = float(env_llm_temp)
        if env_llm_tokens:
            gen_params["max_tokens"] = int(env_llm_tokens)

    return cfg


def create_app(
    pipeline: Optional[IngestionPipeline] = None,
    retrieval_pipeline: Optional[RetrievalPipeline] = None,
    rag_pipeline: Optional[EndToEndRAGPipeline] = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        closed_ids = set()
        for resource in (active_ingest.store, active_retrieval.vector_store):
            if id(resource) in closed_ids:
                continue
            closed_ids.add(id(resource))
            close = getattr(resource, "close", None)
            if callable(close):
                close()

    app = FastAPI(
        title="Enterprise RAG Engine - End-to-End API",
        version="3.0.0",
        description="Modular REST API for ingestion, hybrid retrieval, quantitative evaluation, and LLM generation.",
        lifespan=lifespan,
    )

    cfg = load_default_config()
    active_ingest = pipeline or IngestionPipeline(cfg)
    active_retrieval = retrieval_pipeline or RetrievalPipeline(
        config=cfg,
        document_store=active_ingest.store,
    )
    active_rag = rag_pipeline or EndToEndRAGPipeline(
        config=cfg,
        retrieval_pipeline=active_retrieval,
    )

    cors_origins = [
        origin.strip()
        for origin in os.getenv("RAG_CORS_ORIGINS", "").split(",")
        if origin.strip()
    ]
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["Accept", "Content-Type", "X-API-Key"],
        )

    api_key = os.getenv("RAG_API_KEY", "").strip()

    @app.middleware("http")
    async def require_api_key(request: Request, call_next):
        public_paths = {"/", "/health", "/docs", "/openapi.json", "/redoc"}
        if (
            api_key
            and request.method != "OPTIONS"
            and request.url.path not in public_paths
            and not secrets.compare_digest(request.headers.get("X-API-Key", ""), api_key)
        ):
            return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key."})
        return await call_next(request)

    app.state.pipeline = active_ingest
    app.state.retrieval_pipeline = active_retrieval
    app.state.rag_pipeline = active_rag

    @app.get("/", tags=["System"])
    def api_info():
        return {
            "service": "rag-engine",
            "version": app.version,
            "status": "running",
            "health": "/health",
            "docs": "/docs",
        }

    app.include_router(health_router.router)
    app.include_router(documents_router.router)
    app.include_router(chunks_router.router)
    app.include_router(retrieval_router.router)
    app.include_router(generation_router.router)

    return app


app = create_app()
