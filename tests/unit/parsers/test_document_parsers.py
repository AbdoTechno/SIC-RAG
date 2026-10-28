from io import BytesIO

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from core.registry import registry
import components.parsers


def _text_pdf_bytes(text: str) -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)

    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font)
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
    )

    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(stream)

    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def test_lightweight_pdf_parser_extracts_text_and_page_metadata():
    parser = registry.get_parser(".pdf")()

    documents = parser.parse(
        raw_bytes=_text_pdf_bytes("Pulmora RAG deployment"),
        base_metadata={"source_name": "Deployment Guide", "version_date": "2026-10"},
        file_id="pdf-file-1",
    )

    assert len(documents) == 1
    assert "Pulmora RAG deployment" in documents[0].content
    assert documents[0].file_id == "pdf-file-1"
    assert documents[0].metadata.page_or_section == "Page 1"
    assert documents[0].metadata.extra["page_number"] == 1
