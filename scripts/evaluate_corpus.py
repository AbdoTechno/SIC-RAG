import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "rag_engine"))

from core.config import load_default_config
from pipeline.ingestion_pipeline import IngestionPipeline
from pipeline.retrieval_pipeline import RetrievalPipeline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=Path("data/pulmora_rag.db"))
    parser.add_argument("--dataset", type=Path, default=Path("evaluations/datasets/pulmora_corpus_eval_v1.json"))
    parser.add_argument("--report", type=Path, default=Path("evaluations/history/pulmora_corpus_eval_report_v1.json"))
    args = parser.parse_args()

    cfg = load_default_config()
    cfg["document_store"] = {"provider": "sqlite_store", "params": {"dsn": str(args.db)}}
    cfg["retrieval"]["vector_store"] = {"provider": "sqlite_vector_store", "params": {"dsn": str(args.db), "dimension": 384}}
    ingest = IngestionPipeline(cfg)
    retrieval = RetrievalPipeline(cfg, ingest.store)
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    results = []
    try:
        for case in dataset["cases"]:
            response = retrieval.retrieve(case["query"], top_k=5)
            passages = response.ranked_chunks
            source_match = any(
                case.get("expected_source_contains", "").lower() in chunk.metadata.source_name.lower()
                for chunk in passages
            ) if case.get("expected_source_contains") else True
            page_match = any(
                chunk.metadata.extra.get("pdf_page") in case.get("expected_pdf_pages", [])
                for chunk in passages
            ) if case.get("expected_pdf_pages") else True
            combined_text = " ".join(chunk.text.lower() for chunk in passages)
            term_match = all(term.lower() in combined_text for term in case.get("expected_terms", []))
            passed = (
                response.abstained == case["should_abstain"]
                and (case["should_abstain"] or (source_match and page_match and term_match))
            )
            results.append({
                "id": case["id"], "type": case["type"], "passed": passed,
                "abstained": response.abstained, "abstention_reason": response.abstention_reason,
                "max_score": response.max_confidence_score,
                "source_match": source_match, "page_match": page_match, "term_match": term_match,
                "top_results": [
                    {
                        "source_name": chunk.metadata.source_name,
                        "section": chunk.metadata.page_or_section,
                        "pdf_page": chunk.metadata.extra.get("pdf_page"),
                        "chunk_id": chunk.chunk_id,
                        "rerank_score": chunk.rerank_score,
                    }
                    for chunk in passages
                ],
            })
    finally:
        retrieval.vector_store.close()
        ingest.store.conn.close()

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "total": len(results), "passed": sum(item["passed"] for item in results),
        "pass_rate": round(sum(item["passed"] for item in results) / max(1, len(results)), 4),
        "results": results,
        "note": "Automated retrieval checks are not clinical validation.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
