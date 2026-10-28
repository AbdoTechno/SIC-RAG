import os

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status, Response
from api.dependencies import get_pipeline, get_retrieval_pipeline
from pipeline.ingestion_pipeline import IngestionPipeline
from pipeline.retrieval_pipeline import RetrievalPipeline

router = APIRouter(prefix="/documents", tags=["Documents Management"])


@router.post("")
async def upload_document(
    response: Response,
    file: UploadFile = File(...),
    source_name: str = Form(...),
    version_date: str = Form("N/A"),
    source_link: str = Form("db://raw_files"),
    auto_chunk: bool = Form(True),
    pipeline: IngestionPipeline = Depends(get_pipeline),
    retrieval_pipeline: RetrievalPipeline = Depends(get_retrieval_pipeline),
):
    raw_bytes = await file.read()
    max_upload_mb = int(os.getenv("RAG_MAX_UPLOAD_MB", "20"))
    if len(raw_bytes) > max_upload_mb * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File exceeds the {max_upload_mb} MB upload limit.",
        )
    try:
        upload_res = pipeline.upload_raw_document(
            filename=file.filename or "uploaded.txt",
            raw_bytes=raw_bytes,
            source_name=source_name,
            version_date=version_date,
            source_link=source_link,
        )
        
        # SWE Decision: Dynamic HTTP Status Code based on Idempotency
        if upload_res.get("deduplicated"):
            response.status_code = status.HTTP_200_OK
        else:
            response.status_code = status.HTTP_201_CREATED

        chunk_info = None
        if auto_chunk:
            chunk_res = pipeline.run_chunking_for_file(upload_res["file_id"], force_rechunk=False)
            if chunk_res["status"] == "newly_chunked":
                retrieval_pipeline.invalidate_file_index(upload_res["file_id"])
            chunk_info = {
                "chunking_status": chunk_res["status"],
                "config_hash": chunk_res["config_hash"],
                "chunk_count": chunk_res["chunk_count"],
            }
        return {**upload_res, "chunking": chunk_info}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("", status_code=status.HTTP_200_OK)
def list_documents(
    pipeline: IngestionPipeline = Depends(get_pipeline),
    retrieval_pipeline: RetrievalPipeline = Depends(get_retrieval_pipeline),
):
    index_by_id = {
        item["file_id"]: item
        for item in retrieval_pipeline.knowledge_status()["documents"]
    }
    documents = []
    for document in pipeline.store.list_raw_files():
        index = index_by_id.get(document["file_id"], {})
        documents.append({
            **document,
            "indexed_chunk_count": index.get("indexed_chunk_count", 0),
            "index_status": index.get("index_status", "awaiting_index"),
            "index_error": index.get("error"),
        })
    return {"documents": documents}


@router.get("/{file_id}", status_code=status.HTTP_200_OK)
def get_document_details(
    file_id: str,
    pipeline: IngestionPipeline = Depends(get_pipeline),
    retrieval_pipeline: RetrievalPipeline = Depends(get_retrieval_pipeline),
):
    raw_file = pipeline.store.get_raw_file(file_id)
    if not raw_file:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document '{file_id}' not found.")

    parsed_docs = pipeline.store.get_parsed_documents_by_file(file_id)
    chunks = pipeline.store.get_chunks_by_file(file_id)

    return {
        "file_id": raw_file.file_id,
        "filename": raw_file.filename,
        "file_format": raw_file.file_format,
        "content_hash": raw_file.content_hash,
        "size_bytes": len(raw_file.raw_bytes),
        "metadata": raw_file.metadata,
        "parsed_documents_count": len(parsed_docs),
        "chunks_count": len(chunks),
        "indexed_chunks_count": retrieval_pipeline.vector_store.count_vectors(file_id),
        "index_status": next(
            (
                item["index_status"]
                for item in retrieval_pipeline.knowledge_status()["documents"]
                if item["file_id"] == file_id
            ),
            "awaiting_index",
        ),
    }


@router.delete("/{file_id}", status_code=status.HTTP_200_OK)
def delete_document(
    file_id: str,
    pipeline: IngestionPipeline = Depends(get_pipeline),
    retrieval_pipeline: RetrievalPipeline = Depends(get_retrieval_pipeline),
):
    if not pipeline.store.get_raw_file(file_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document '{file_id}' not found.")
    retrieval_pipeline.vector_store.delete_by_file_id(file_id)
    deleted = pipeline.store.delete_file(file_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document '{file_id}' not found.")
    return {"status": "deleted", "file_id": file_id}
