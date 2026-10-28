"""Verify durable retrieval and the typed case endpoint against an ingested SQLite corpus."""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "rag_engine"))

from core.config import load_default_config
from core.registry import registry
from pipeline.ingestion_pipeline import IngestionPipeline
from pipeline.rag_pipeline import EndToEndRAGPipeline
from pipeline.retrieval_pipeline import RetrievalPipeline


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=Path("data/pulmora_rag.db"))
    parser.add_argument("--fixture", type=Path, default=Path("examples/case_review_request.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/e2e_verification.json"))
    return parser.parse_args()


def _components(db_path: Path):
    cfg = load_default_config()
    cfg["document_store"] = {"provider": "sqlite_store", "params": {"dsn": str(db_path)}}
    cfg["retrieval"]["vector_store"] = {
        "provider": "sqlite_vector_store",
        "params": {"dsn": str(db_path), "dimension": cfg["retrieval"]["embedder"]["params"]["dim"]},
    }
    ingest = IngestionPipeline(cfg)
    retrieval = RetrievalPipeline(cfg, ingest.store)
    generator = registry.get_generator("extractive_mock_generator")()
    rag = EndToEndRAGPipeline(cfg, retrieval, generator=generator)
    return ingest, retrieval, rag


def _close(ingest, retrieval) -> None:
    for resource in (retrieval.vector_store, ingest.store):
        close = getattr(resource, "close", None)
        if callable(close):
            close()
        elif hasattr(resource, "conn"):
            resource.conn.close()


def main() -> int:
    args = _args()
    if not args.db.is_file() or not args.fixture.is_file():
        print("Ingested database or synthetic fixture is missing.", file=sys.stderr)
        return 2

    # api.main constructs its module-level ASGI app during import. Pin it to the
    # same local durable store so importing the factory never contacts production.
    os.environ["RAG_STORE_PROVIDER"] = "sqlite_store"
    os.environ["RAG_DB_DSN"] = str(args.db)
    from api.main import create_app

    first_ingest, first_retrieval, _ = _components(args.db)
    before = first_retrieval.retrieve("How is hospital-acquired pneumonia defined?", top_k=3)
    if before.abstained or not before.ranked_chunks:
        print("Pre-restart retrieval failed.", file=sys.stderr)
        return 1
    first_chunk = before.ranked_chunks[0].chunk_id
    _close(first_ingest, first_retrieval)

    ingest, retrieval, rag = _components(args.db)
    after = retrieval.retrieve("How is hospital-acquired pneumonia defined?", top_k=3)
    restart_passed = not after.abstained and any(chunk.chunk_id == first_chunk for chunk in after.ranked_chunks)
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    headers = {"X-Request-ID": fixture["request_id"]}
    configured_api_key = os.getenv("RAG_API_KEY", "").strip()
    if configured_api_key:
        headers["X-API-Key"] = configured_api_key

    app = create_app(ingest, retrieval, rag)
    with TestClient(app) as client:
        knowledge_response = client.get("/knowledge/status", headers=headers)
        case_response = client.post("/rag/case-review", json=fixture, headers=headers)

    if knowledge_response.status_code != 200 or case_response.status_code != 200:
        print("Readiness or case endpoint failed.", file=sys.stderr)
        return 1
    knowledge = knowledge_response.json()
    case = case_response.json()
    correlations_passed = (
        case["request_id"] == fixture["request_id"]
        and case["assessment_id"] == fixture["assessment_id"]
        and case["input_revision"] == fixture["input_revision"]
    )
    citations_resolve = bool(case["sources_cited"]) and all(
        source.get("file_id") and source.get("chunk_id") and source.get("pdf_page")
        for source in case["sources_cited"]
    )
    passed = bool(
        restart_passed and knowledge["retrieval_ready"] and correlations_passed
        and not case["abstained"] and citations_resolve
    )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "generation_mode": "extractive_mock_generator",
        "restart_retrieval_passed": restart_passed,
        "knowledge": {
            "retrieval_ready": knowledge["retrieval_ready"],
            "registered_document_count": knowledge["registered_document_count"],
            "parsed_chunk_count": knowledge["parsed_chunk_count"],
            "indexed_chunk_count": knowledge["indexed_chunk_count"],
        },
        "case": {
            "http_status": case_response.status_code,
            "correlations_passed": correlations_passed,
            "abstained": case["abstained"],
            "citations_resolve": citations_resolve,
            "source_count": len(case["sources_cited"]),
            "source_pages": [source.get("pdf_page") for source in case["sources_cited"]],
        },
        "note": "Automated engineering verification with a deterministic mock generator; not clinical validation.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
