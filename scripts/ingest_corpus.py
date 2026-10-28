"""Repeatable, idempotent ingestion for the Pulmora reference corpus."""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "rag_engine"))

from core.config import load_default_config
from pipeline.ingestion_pipeline import IngestionPipeline
from pipeline.retrieval_pipeline import RetrievalPipeline


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest and index Pulmora reference PDFs.")
    parser.add_argument("--pdf-dir", type=Path, default=Path("pdfs"))
    parser.add_argument("--manifest", type=Path, default=Path("pdfs/sources.json"))
    parser.add_argument("--db", type=Path, default=Path("data/pulmora_rag.db"))
    parser.add_argument("--report", type=Path, default=Path("reports/ingestion_report.json"))
    parser.add_argument("--include-optional", action="store_true")
    return parser.parse_args()


def _inspect_pdf(path: Path) -> Dict[str, Any]:
    reader = PdfReader(path)
    page_characters = []
    empty_pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").replace("\x00", "").strip()
        page_characters.append(len(text))
        if not text:
            empty_pages.append(number)
    return {
        "page_count": len(reader.pages),
        "pages_extracted": len(reader.pages) - len(empty_pages),
        "empty_pages": empty_pages,
        "sparse_pages": [index + 1 for index, count in enumerate(page_characters) if 0 < count < 200],
        "text_characters": sum(page_characters),
    }


def main() -> int:
    args = _args()
    if not args.pdf_dir.is_dir() or not args.manifest.is_file():
        print("Corpus folder or sources.json is missing.", file=sys.stderr)
        return 2

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    documents = sorted(manifest.get("documents", []), key=lambda item: item["filename"])
    args.db.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    cfg = load_default_config()
    cfg["document_store"] = {"provider": "sqlite_store", "params": {"dsn": str(args.db)}}
    cfg.setdefault("retrieval", {})["vector_store"] = {
        "provider": "sqlite_vector_store",
        "params": {"dsn": str(args.db), "dimension": cfg["retrieval"]["embedder"]["params"]["dim"]},
    }
    ingest = IngestionPipeline(cfg)
    retrieval = RetrievalPipeline(cfg, ingest.store)
    results = []
    failures = 0

    try:
        for entry in documents:
            filename = entry["filename"]
            path = args.pdf_dir / filename
            result: Dict[str, Any] = {"filename": filename, "status": "pending", "warnings": [], "errors": []}
            if not entry.get("ingest_default", False) and not args.include_optional:
                result.update(status="skipped", reason="ingest_default=false; requires layout/scope review")
                results.append(result)
                continue
            try:
                if not path.is_file():
                    raise FileNotFoundError(f"Required PDF is missing: {filename}")
                raw = path.read_bytes()
                digest = hashlib.sha256(raw).hexdigest()
                if digest != entry.get("sha256"):
                    raise ValueError("Content hash does not match sources.json.")
                inspection = _inspect_pdf(path)
                if inspection["page_count"] != entry.get("page_count"):
                    raise ValueError("PDF page count does not match sources.json.")
                if inspection["pages_extracted"] == 0 or inspection["text_characters"] < 200:
                    raise ValueError("PDF has no usable extracted text; OCR review is required.")
                if inspection["empty_pages"]:
                    result["warnings"].append(f"Empty extracted pages: {inspection['empty_pages']}")
                if inspection["sparse_pages"]:
                    excluded = set(entry.get("exclude_pdf_pages", []))
                    reviewed_excluded = [page for page in inspection["sparse_pages"] if page in excluded]
                    unresolved_sparse = [page for page in inspection["sparse_pages"] if page not in excluded]
                    if reviewed_excluded:
                        result["warnings"].append(f"Sparse pages excluded after visual review: {reviewed_excluded}")
                    if unresolved_sparse:
                        result["warnings"].append(f"Sparse pages require visual review: {unresolved_sparse}")

                upload = ingest.upload_raw_document(
                    filename=filename,
                    raw_bytes=raw,
                    source_name=entry["document_title"],
                    version_date=entry.get("last_update_verified") or entry.get("publication_date") or "N/A",
                    source_link=entry.get("source_url") or "db://raw_files",
                    extra_metadata={
                        "organization": entry.get("organization"),
                        "publication_date": entry.get("publication_date"),
                        "last_update_verified": entry.get("last_update_verified"),
                        "document_family": entry.get("document_family"),
                        "patient_population": entry.get("patient_population"),
                        "topics": entry.get("topics", []),
                        "language": entry.get("language"),
                        "exclude_pdf_pages": entry.get("exclude_pdf_pages", []),
                    },
                )
                chunked = ingest.run_chunking_for_file(upload["file_id"])
                current_vectors = retrieval.vector_store.count_vectors(upload["file_id"])
                if (
                    chunked["status"] == "cached_skip"
                    and current_vectors == chunked["chunk_count"]
                    and current_vectors > 0
                ):
                    index_result = {"index_status": "already_indexed", "indexed_chunks_count": current_vectors}
                else:
                    index_result = retrieval.index_file_chunks(upload["file_id"])
                result.update(
                    status="ready", file_id=upload["file_id"], content_hash=digest,
                    deduplicated=upload["deduplicated"], pages_extracted=inspection["pages_extracted"],
                    pages_indexed=inspection["pages_extracted"] - len(entry.get("exclude_pdf_pages", [])),
                    excluded_pdf_pages=entry.get("exclude_pdf_pages", []),
                    page_count=inspection["page_count"], chunks_created=chunked["chunk_count"],
                    chunk_status=chunked["status"], index_status=index_result["index_status"],
                    indexed_chunks=index_result["indexed_chunks_count"],
                )
            except Exception as exc:
                failures += 1
                result.update(status="failed")
                result["errors"].append(str(exc)[:500])
            results.append(result)
    finally:
        for resource in (retrieval.vector_store, ingest.store):
            close = getattr(resource, "close", None)
            if callable(close):
                close()

    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "database": str(args.db),
        "page_number_convention": "1-based physical PDF page",
        "include_optional": args.include_optional,
        "summary": {
            "ready": sum(item["status"] == "ready" for item in results),
            "skipped": sum(item["status"] == "skipped" for item in results),
            "failed": failures,
        },
        "documents": results,
    }
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
