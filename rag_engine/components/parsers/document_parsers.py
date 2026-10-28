from io import BytesIO
import json
from typing import List, Dict, Any, Optional
from pypdf import PdfReader
from core.interfaces import BaseFileParser
from core.schemas import RawDocument, SourceMetadata
from core.registry import registry


@registry.register_parser(".json")
class JSONRecordParser(BaseFileParser):
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
                source_name=item.get("source_name", base_metadata.get("source_name", "JSON Document")),
                version_date=str(item.get("version_date", base_metadata.get("version_date", "N/A"))),
                source_link=item.get("source_link", base_metadata.get("source_link", "postgres://raw_files")),
                file_format="json",
                page_or_section=item.get("section", f"Record_{idx}"),
                extra=item.get("extra", {}),
            )
            raw_docs.append(RawDocument(content=content, metadata=metadata, file_id=file_id))

        if not raw_docs:
            raise ValueError("No valid records found in JSON payload.")
        return raw_docs


# DRY - base class handles common UTF-8 decoding logic for multiple text formats
class BaseUTF8TextParser(BaseFileParser):
    def __init__(self, format_label: str):
        self.format_label = format_label

    def parse(self, raw_bytes: bytes, base_metadata: Dict[str, Any], file_id: Optional[str] = None) -> List[RawDocument]:
        content = raw_bytes.decode("utf-8").strip()
        if not content:
            raise ValueError(f"Empty {self.format_label} byte stream.")

        metadata = SourceMetadata(
            source_name=base_metadata.get("source_name", "Text Document"),
            version_date=str(base_metadata.get("version_date", "N/A")),
            source_link=base_metadata.get("source_link", "postgres://raw_files"),
            file_format=self.format_label,
            page_or_section=base_metadata.get("section", "Full Document"),
            extra=base_metadata.get("extra", {}),
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
    def parse(
        self,
        raw_bytes: bytes,
        base_metadata: Dict[str, Any],
        file_id: Optional[str] = None
    ) -> List[RawDocument]:
        if not raw_bytes:
            raise ValueError("PDF byte stream cannot be empty.")

        raw_docs: List[RawDocument] = []
        try:
            reader = PdfReader(BytesIO(raw_bytes))
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception as exc:
                    raise ValueError("Encrypted PDF files are not supported.") from exc

            excluded_pages = {
                int(number)
                for number in base_metadata.get("extra", {}).get("exclude_pdf_pages", [])
            }
            for page_idx, page in enumerate(reader.pages):
                physical_page = page_idx + 1
                if physical_page in excluded_pages:
                    continue
                page_text = (page.extract_text() or "").strip().replace("\x00", "")
                if not page_text:
                    continue

                meta = SourceMetadata(
                    source_name=base_metadata.get("source_name", "PDF Document"),
                    version_date=str(base_metadata.get("version_date", "N/A")),
                    source_link=base_metadata.get("source_link", "postgres://raw_files"),
                    file_format="pdf",
                    page_or_section=f"Page {physical_page}",
                    extra={
                        **base_metadata.get("extra", {}),
                        "page_number": physical_page,
                        "pdf_page": physical_page,
                        "pdf_page_convention": "1-based physical PDF page",
                    },
                )
                raw_docs.append(
                    RawDocument(
                        content=page_text,
                        metadata=meta,
                        file_id=file_id,
                    )
                )
        except Exception as e:
            raise ValueError(f"Failed to parse PDF binary stream: {str(e)}")

        if not raw_docs:
            raise ValueError("PDF contained no extractable textual content.")

        return raw_docs
