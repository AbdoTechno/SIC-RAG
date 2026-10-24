import json
import sqlite3
from typing import List, Optional
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from core.interfaces import BaseDocumentStore
from core.schemas import StoredRawFile, RawDocument, SourceMetadata
from core.registry import registry


def _metadata_to_dict(meta: SourceMetadata) -> dict:
    return {
        "source_name": meta.source_name,
        "version_date": meta.version_date,
        "source_link": meta.source_link,
        "file_format": meta.file_format,
        "page_or_section": meta.page_or_section,
        "extra": meta.extra,
    }


def _dict_to_metadata(data: dict) -> SourceMetadata:
    return SourceMetadata(
        source_name=data["source_name"],
        version_date=data["version_date"],
        source_link=data["source_link"],
        file_format=data["file_format"],
        page_or_section=data.get("page_or_section"),
        extra=data.get("extra", {}),
    )


@registry.register_document_store("postgres_store")
class PostgresDocumentStore(BaseDocumentStore):
    """
    Production PostgreSQL Document & Blob Store.
    Uses BYTEA for raw clinical files, JSONB for indexed metadata,
    and atomic ON CONFLICT for SHA-256 deduplication.
    """

    def __init__(self, dsn: str):
        self.dsn = dsn
        self._create_tables()

    def _get_connection(self):
        return psycopg.connect(self.dsn, row_factory=dict_row)

    def _create_tables(self) -> None:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS raw_files (
                        file_id TEXT PRIMARY KEY,
                        filename TEXT NOT NULL,
                        file_format TEXT NOT NULL,
                        content_hash TEXT UNIQUE NOT NULL,
                        raw_bytes BYTEA NOT NULL,
                        metadata JSONB NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_raw_files_content_hash 
                    ON raw_files(content_hash);
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS parsed_documents (
                        doc_id TEXT PRIMARY KEY,
                        file_id TEXT NOT NULL REFERENCES raw_files(file_id) ON DELETE CASCADE,
                        content TEXT NOT NULL,
                        metadata JSONB NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_parsed_docs_file_id 
                    ON parsed_documents(file_id);
                """)
            conn.commit()

    def save_raw_file(self, raw_file: StoredRawFile) -> str:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Atomic upsert check via SHA-256 content_hash
                cur.execute(
                    """
                    INSERT INTO raw_files (file_id, filename, file_format, content_hash, raw_bytes, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (content_hash) DO UPDATE 
                    SET content_hash = EXCLUDED.content_hash
                    RETURNING file_id;
                    """,
                    (
                        raw_file.file_id,
                        raw_file.filename,
                        raw_file.file_format,
                        raw_file.content_hash,
                        raw_file.raw_bytes,
                        Jsonb(_metadata_to_dict(raw_file.metadata)),
                    ),
                )
                row = cur.fetchone()
            conn.commit()
            return row["file_id"]

    def get_raw_file(self, file_id: str) -> Optional[StoredRawFile]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT file_id, filename, file_format, raw_bytes, metadata
                    FROM raw_files
                    WHERE file_id = %s;
                    """,
                    (file_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None

                return StoredRawFile(
                    file_id=row["file_id"],
                    filename=row["filename"],
                    file_format=row["file_format"],
                    raw_bytes=bytes(row["raw_bytes"]),
                    metadata=_dict_to_metadata(row["metadata"]),
                )

    def save_parsed_documents(self, documents: List[RawDocument]) -> None:
        if not documents:
            return
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                for doc in documents:
                    cur.execute(
                        """
                        INSERT INTO parsed_documents (doc_id, file_id, content, metadata)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (doc_id) DO UPDATE
                        SET content = EXCLUDED.content,
                            metadata = EXCLUDED.metadata;
                        """,
                        (
                            doc.doc_id,
                            doc.file_id,
                            doc.content,
                            Jsonb(_metadata_to_dict(doc.metadata)),
                        ),
                    )
            conn.commit()

    def get_parsed_documents_by_file(self, file_id: str) -> List[RawDocument]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT doc_id, file_id, content, metadata
                    FROM parsed_documents
                    WHERE file_id = %s
                    ORDER BY created_at ASC;
                    """,
                    (file_id,),
                )
                rows = cur.fetchall()
                return [
                    RawDocument(
                        doc_id=r["doc_id"],
                        file_id=r["file_id"],
                        content=r["content"],
                        metadata=_dict_to_metadata(r["metadata"]),
                    )
                    for r in rows
                ]


@registry.register_document_store("sqlite_store")
class SQLiteDocumentStore(BaseDocumentStore):
    """
    Lightweight fallback / CI test store implementing the exact same BaseDocumentStore contract.
    """

    def __init__(self, dsn: str = ":memory:"):
        self.dsn = dsn
        self.conn = sqlite3.connect(self.dsn, check_same_thread=False)
        self._create_tables()

    def _create_tables(self) -> None:
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS raw_files (
                file_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_format TEXT NOT NULL,
                content_hash TEXT UNIQUE NOT NULL,
                raw_bytes BLOB NOT NULL,
                metadata TEXT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS parsed_documents (
                doc_id TEXT PRIMARY KEY,
                file_id TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def save_raw_file(self, raw_file: StoredRawFile) -> str:
        cur = self.conn.cursor()
        cur.execute("SELECT file_id FROM raw_files WHERE content_hash = ?", (raw_file.content_hash,))
        existing = cur.fetchone()
        if existing:
            return existing[0]

        cur.execute(
            "INSERT INTO raw_files VALUES (?, ?, ?, ?, ?, ?)",
            (
                raw_file.file_id,
                raw_file.filename,
                raw_file.file_format,
                raw_file.content_hash,
                raw_file.raw_bytes,
                json.dumps(_metadata_to_dict(raw_file.metadata)),
            ),
        )
        self.conn.commit()
        return raw_file.file_id

    def get_raw_file(self, file_id: str) -> Optional[StoredRawFile]:
        cur = self.conn.cursor()
        cur.execute("SELECT file_id, filename, file_format, raw_bytes, metadata FROM raw_files WHERE file_id = ?", (file_id,))
        row = cur.fetchone()
        if not row:
            return None
        return StoredRawFile(
            file_id=row[0],
            filename=row[1],
            file_format=row[2],
            raw_bytes=row[3],
            metadata=_dict_to_metadata(json.loads(row[4])),
        )

    def save_parsed_documents(self, documents: List[RawDocument]) -> None:
        cur = self.conn.cursor()
        for doc in documents:
            cur.execute(
                "INSERT OR REPLACE INTO parsed_documents VALUES (?, ?, ?, ?)",
                (doc.doc_id, doc.file_id or "unlinked", doc.content, json.dumps(_metadata_to_dict(doc.metadata))),
            )
        self.conn.commit()

    def get_parsed_documents_by_file(self, file_id: str) -> List[RawDocument]:
        cur = self.conn.cursor()
        cur.execute("SELECT doc_id, file_id, content, metadata FROM parsed_documents WHERE file_id = ?", (file_id,))
        return [
            RawDocument(doc_id=r[0], file_id=r[1], content=r[2], metadata=_dict_to_metadata(json.loads(r[3])))
            for r in cur.fetchall()
        ]