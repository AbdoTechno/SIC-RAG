import os
import secrets
import uuid
from contextlib import asynccontextmanager
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pipeline.ingestion_pipeline import IngestionPipeline
from pipeline.retrieval_pipeline import RetrievalPipeline
from pipeline.rag_pipeline import EndToEndRAGPipeline
from core.config import load_default_config
from api.errors import APIError, error_body

from api.routers import (
    health_router,
    documents_router,
    chunks_router,
    retrieval_router,
    generation_router,
    knowledge_router,
)

load_dotenv()


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
        version="3.1.0",
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
        request_id = request.headers.get("X-Request-ID", "").strip()[:128] or str(uuid.uuid4())
        request.state.request_id = request_id
        public_paths = {"/", "/health", "/docs", "/openapi.json", "/redoc"}
        if (
            api_key
            and request.method != "OPTIONS"
            and request.url.path not in public_paths
            and not secrets.compare_digest(request.headers.get("X-API-Key", ""), api_key)
        ):
            return JSONResponse(
                status_code=401,
                content=error_body("UNAUTHORIZED", "Invalid or missing API key.", request_id),
                headers={"X-Request-ID": request_id},
            )
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(APIError)
    async def handle_api_error(request: Request, exc: APIError):
        request_id = exc.request_id or getattr(request.state, "request_id", str(uuid.uuid4()))
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(exc.code, exc.message, request_id, exc.retryable),
            headers={"X-Request-ID": request_id},
        )

    @app.exception_handler(HTTPException)
    async def handle_http_error(request: Request, exc: HTTPException):
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        message = str(exc.detail)
        codes = {400: "INVALID_OPERATION", 404: "NOT_FOUND", 413: "PAYLOAD_TOO_LARGE", 503: "DEPENDENCY_UNAVAILABLE"}
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(codes.get(exc.status_code, "HTTP_ERROR"), message, request_id, exc.status_code >= 500),
            headers={"X-Request-ID": request_id},
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError):
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        return JSONResponse(
            status_code=422,
            content={
                **error_body("VALIDATION_ERROR", "Request validation failed.", request_id),
                "validation_errors": exc.errors(),
            },
            headers={"X-Request-ID": request_id},
        )

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
    app.include_router(knowledge_router.router)

    return app


app = create_app()
