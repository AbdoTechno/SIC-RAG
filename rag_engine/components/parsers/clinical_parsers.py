import json
from typing import List, Dict, Any, Optional
from core.interfaces import BaseFileParser
from core.schemas import RawDocument, SourceMetadata
from core.registry import registry

@registry.register_parser(".json")
class JSONClinicalParser(BaseFileParser):
    def parse(self, raw_bytes: bytes, base_metadata: Dict[str, Any], file_id: Optional[str] = None) -> List[RawDocument]:
        if not raw_bytes or not raw_bytes.strip():
            raise ValueError("JSON raw_bytes cannot be empty.")

        data = json.loads(raw_bytes.decode("utf-8"))
        records = data if isinstance(data, list) else [data]
        raw_docs: List[RawDocument] = []

        for idx, item in enumerate(records):
            content = item.get("text") or item.get("content") or json.dumps(item, ensure_ascii=False)
            if not content.strip():
                continue

            metadata = SourceMetadata(
                source_name=item.get("source_name", base_metadata.get("source_name", "Clinical JSON Guideline")),
                version_date=str(item.get("version_date", base_metadata.get("version_date", "N/A"))),
                source_link=item.get("source_link", base_metadata.get("source_link", "db://raw_files")),
                file_format="json",
                page_or_section=item.get("section", f"Record_{idx}"),
                extra=item.get("extra", {})
            )
            raw_docs.append(RawDocument(content=content, metadata=metadata, file_id=file_id))

        if not raw_docs:
            raise ValueError("No valid records found in JSON payload.")
        return raw_docs

class BaseUTF8TextParser(BaseFileParser):
    def __init__(self, format_label: str):
        self.format_label = format_label

    def parse(self, raw_bytes: bytes, base_metadata: Dict[str, Any], file_id: Optional[str] = None) -> List[RawDocument]:
        content = raw_bytes.decode("utf-8").strip()
        if not content:
            raise ValueError(f"Empty {self.format_label} byte stream.")

        metadata = SourceMetadata(
            source_name=base_metadata.get("source_name", "Clinical Document"),
            version_date=str(base_metadata.get("version_date", "N/A")),
            source_link=base_metadata.get("source_link", "db://raw_files"),
            file_format=self.format_label,
            page_or_section=base_metadata.get("section", "Full Document"),
            extra=base_metadata.get("extra", {})
        )
        return [RawDocument(content=content, metadata=metadata, file_id=file_id)]

@registry.register_parser(".md")
class MarkdownParser(BaseUTF8TextParser):
    def __init__(self):
        super().__init__(format_label="markdown")

@registry.register_parser(".txt")
class PlainTextParser(BaseUTF8TextParser):
    def __init__(self):
        super().__init__(format_label="text")

@registry.register_parser(".pdf")
class PDFStreamParser(BaseFileParser):
    def parse(self, raw_bytes: bytes, base_metadata: Dict[str, Any], file_id: Optional[str] = None) -> List[RawDocument]:
        if not raw_bytes:
            raise ValueError("PDF byte stream cannot be empty.")

        # Stream-compatible contract for PyMuPDF (fitz.open(stream=raw_bytes, filetype="pdf")) or Docling DocumentStream
        simulated_pages = [
            {"page": 1, "text": raw_bytes.decode("utf-8", errors="ignore") or "Extracted PDF clinical content."}
        ]

        docs: List[RawDocument] = []
        for p in simulated_pages:
            meta = SourceMetadata(
                source_name=base_metadata.get("source_name", "PDF Guideline"),
                version_date=str(base_metadata.get("version_date", "2025-01")),
                source_link=base_metadata.get("source_link", "db://raw_files"),
                file_format="pdf",
                page_or_section=f"Page {p['page']}",
                extra=base_metadata.get("extra", {})
            )
            docs.append(RawDocument(content=p["text"], metadata=meta, file_id=file_id))
        return docs