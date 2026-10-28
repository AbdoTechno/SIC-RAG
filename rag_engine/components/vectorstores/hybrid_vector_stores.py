import math
import re
import json
import sqlite3
from dataclasses import replace
from typing import List, Dict, Optional
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from core.interfaces import BaseVectorStore
from core.schemas import DocumentChunk, SourceMetadata
from core.registry import registry


def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return dot / (n1 * n2)


def _lexical_overlap_score(query: str, text: str) -> float:
    stopwords = {
        "what", "is", "are", "how", "does", "do", "the", "a", "an", "in", "on", "of", "to",
        "for", "and", "or", "with", "by", "from", "about", "which", "relevant", "reference",
        "evidence", "clinical", "patient", "assessment", "treatment", "treat", "discussed",
    }
    q_tokens = {
        token for token in re.findall(r"[a-z0-9\u0600-\u06ff]+(?:-[a-z0-9\u0600-\u06ff]+)*", query.lower())
        if token not in stopwords
    }
    if not q_tokens:
        return 0.0
    t_lower = text.lower()
    t_tokens = set(re.findall(r"[a-z0-9\u0600-\u06ff]+(?:-[a-z0-9\u0600-\u06ff]+)*", t_lower))

    exact_matches = sum(1 for qt in q_tokens if qt in t_tokens or qt in t_lower)
    return exact_matches / len(q_tokens)


def _apply_rrf_fusion(
    chunks: List[DocumentChunk],
    semantic_scores: Dict[str, float],
    lexical_scores: Dict[str, float],
    mode: str = "hybrid",
    rrf_k: int = 60,
) -> List[DocumentChunk]:
    if not chunks:
        return []

    sem_sorted = sorted(chunks, key=lambda c: semantic_scores.get(c.chunk_id, 0.0), reverse=True)
    lex_sorted = sorted(chunks, key=lambda c: lexical_scores.get(c.chunk_id, 0.0), reverse=True)

    sem_ranks = {c.chunk_id: rank + 1 for rank, c in enumerate(sem_sorted)}
    lex_ranks = {c.chunk_id: rank + 1 for rank, c in enumerate(lex_sorted)}

    scored_chunks: List[DocumentChunk] = []
    for c in chunks:
        cid = c.chunk_id
        s_score = round(semantic_scores.get(cid, 0.0), 4)
        l_score = round(lexical_scores.get(cid, 0.0), 4)

        if mode == "semantic":
            combined = s_score
        elif mode == "lexical":
            combined = l_score
        else:
            # Reciprocal Rank Fusion (RRF) with lexical boost when exact terms match
            rrf_sem = 1.0 / (rrf_k + sem_ranks[cid])
            rrf_lex = (1.0 / (rrf_k + lex_ranks[cid])) if l_score > 0 else 0.0
            combined = round((rrf_sem + rrf_lex) * 30.0 + (0.35 * s_score) + (0.35 * l_score), 4)

        scored_chunks.append(
            replace(
                c,
                similarity_score=s_score,
                lexical_score=l_score,
                rrf_score=combined,
            )
        )

    scored_chunks.sort(key=lambda x: x.rrf_score or 0.0, reverse=True)
    return scored_chunks


@registry.register_vector_store("in_memory_vector_store")
class InMemoryHybridVectorStore(BaseVectorStore):
    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k
        self._store: Dict[str, DocumentChunk] = {}
        self._statuses: Dict[str, Dict[str, object]] = {}

    def upsert_chunks(self, chunks: List[DocumentChunk]) -> int:
        count = 0
        for chunk in chunks:
            if not chunk.embedding:
                raise ValueError(f"Chunk '{chunk.chunk_id}' is missing its embedding vector.")
            self._store[chunk.chunk_id] = chunk
            count += 1
        return count

    def search(
        self,
        query_embedding: List[float],
        query_text: str,
        top_k: int = 5,
        mode: str = "hybrid",
        file_id_filter: Optional[str] = None,
        content_type_filter: Optional[str] = None,
    ) -> List[DocumentChunk]:
        candidates = list(self._store.values())
        if file_id_filter:
            candidates = [c for c in candidates if c.file_id == file_id_filter]
        if content_type_filter:
            candidates = [c for c in candidates if c.content_type == content_type_filter]

        if not candidates:
            return []

        sem_scores = {c.chunk_id: _cosine_similarity(query_embedding, c.embedding or []) for c in candidates}
        lex_scores = {c.chunk_id: _lexical_overlap_score(query_text, c.text) for c in candidates}

        fused = _apply_rrf_fusion(candidates, sem_scores, lex_scores, mode=mode, rrf_k=self.rrf_k)
        return fused[:top_k]

    def delete_by_file_id(self, file_id: str) -> int:
        to_delete = [cid for cid, c in self._store.items() if c.file_id == file_id]
        for cid in to_delete:
            del self._store[cid]
        return len(to_delete)

    def count_vectors(self, file_id: Optional[str] = None) -> int:
        if file_id:
            return sum(1 for c in self._store.values() if c.file_id == file_id)
        return len(self._store)

    def set_index_status(self, file_id: str, status: str, indexed_chunks: int = 0, error: Optional[str] = None) -> None:
        self._statuses[file_id] = {"status": status, "indexed_chunks": indexed_chunks, "error": error}

    def get_index_statuses(self) -> Dict[str, Dict[str, object]]:
        return dict(self._statuses)


@registry.register_vector_store("sqlite_vector_store")
class SQLiteHybridVectorStore(BaseVectorStore):
    """Durable exact-search vector store for single-instance SQLite deployments."""

    def __init__(self, dsn: str, dimension: int = 384, rrf_k: int = 60):
        self.dsn = dsn
        self.dimension = dimension
        self.rrf_k = rrf_k
        self.conn = sqlite3.connect(dsn, check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS chunk_vectors (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                file_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content_type TEXT NOT NULL,
                text TEXT NOT NULL,
                metadata TEXT NOT NULL,
                embedding TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_sqlite_vectors_file ON chunk_vectors(file_id);
            CREATE TABLE IF NOT EXISTS document_index_status (
                file_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                indexed_chunks INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def upsert_chunks(self, chunks: List[DocumentChunk]) -> int:
        for chunk in chunks:
            if not chunk.embedding or len(chunk.embedding) != self.dimension:
                raise ValueError(f"Chunk '{chunk.chunk_id}' has an invalid embedding dimension.")
            metadata = {
                "source_name": chunk.metadata.source_name,
                "version_date": chunk.metadata.version_date,
                "source_link": chunk.metadata.source_link,
                "file_format": chunk.metadata.file_format,
                "page_or_section": chunk.metadata.page_or_section,
                "extra": chunk.metadata.extra,
            }
            self.conn.execute(
                """
                INSERT OR REPLACE INTO chunk_vectors
                (chunk_id, doc_id, file_id, chunk_index, content_type, text, metadata, embedding, dimension, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    chunk.chunk_id, chunk.doc_id, chunk.file_id, chunk.chunk_index,
                    chunk.content_type, chunk.text, json.dumps(metadata),
                    json.dumps(chunk.embedding), self.dimension,
                ),
            )
        self.conn.commit()
        return len(chunks)

    def search(self, query_embedding: List[float], query_text: str, top_k: int = 5,
               mode: str = "hybrid", file_id_filter: Optional[str] = None,
               content_type_filter: Optional[str] = None) -> List[DocumentChunk]:
        clauses, params = ["dimension = ?"], [self.dimension]
        if file_id_filter:
            clauses.append("file_id = ?")
            params.append(file_id_filter)
        if content_type_filter:
            clauses.append("content_type = ?")
            params.append(content_type_filter)
        rows = self.conn.execute(
            f"SELECT chunk_id, doc_id, file_id, chunk_index, content_type, text, metadata, embedding "
            f"FROM chunk_vectors WHERE {' AND '.join(clauses)}",
            tuple(params),
        ).fetchall()
        candidates, sem_scores, lex_scores = [], {}, {}
        for row in rows:
            meta = json.loads(row[6])
            chunk = DocumentChunk(
                chunk_id=row[0], doc_id=row[1], file_id=row[2], chunk_index=row[3],
                content_type=row[4], text=row[5],
                metadata=SourceMetadata(
                    source_name=meta["source_name"], version_date=meta["version_date"],
                    source_link=meta["source_link"], file_format=meta["file_format"],
                    page_or_section=meta.get("page_or_section"), extra=meta.get("extra", {}),
                ),
            )
            candidates.append(chunk)
            sem_scores[chunk.chunk_id] = _cosine_similarity(query_embedding, json.loads(row[7]))
            lex_scores[chunk.chunk_id] = _lexical_overlap_score(query_text, chunk.text)
        return _apply_rrf_fusion(candidates, sem_scores, lex_scores, mode, self.rrf_k)[:top_k]

    def delete_by_file_id(self, file_id: str) -> int:
        cursor = self.conn.execute("DELETE FROM chunk_vectors WHERE file_id = ?", (file_id,))
        self.conn.commit()
        return cursor.rowcount

    def count_vectors(self, file_id: Optional[str] = None) -> int:
        if file_id:
            return int(self.conn.execute("SELECT COUNT(*) FROM chunk_vectors WHERE file_id = ?", (file_id,)).fetchone()[0])
        return int(self.conn.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0])

    def set_index_status(self, file_id: str, status: str, indexed_chunks: int = 0, error: Optional[str] = None) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO document_index_status
               (file_id, status, indexed_chunks, error, updated_at)
               VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)""",
            (file_id, status, indexed_chunks, error),
        )
        self.conn.commit()

    def get_index_statuses(self) -> Dict[str, Dict[str, object]]:
        rows = self.conn.execute(
            "SELECT file_id, status, indexed_chunks, error, updated_at FROM document_index_status"
        ).fetchall()
        return {row[0]: {"status": row[1], "indexed_chunks": row[2], "error": row[3], "updated_at": row[4]} for row in rows}


# System Design Decision: Native PostgreSQL pgvector with HNSW index for scalable ANN search without external infra overhead.
@registry.register_vector_store("pgvector_store")
class PgVectorHybridStore(BaseVectorStore):
    def __init__(
        self,
        dsn: str,
        dimension: int = 384,
        rrf_k: int = 60,
        use_pool: bool = True,
        min_size: int = 1,
        max_size: int = 3,
    ):
        self.dsn = dsn
        self.dimension = dimension
        self.rrf_k = rrf_k
        self._pool: Optional[ConnectionPool] = None
        if use_pool:
            self._pool = ConnectionPool(
                conninfo=self.dsn,
                min_size=min_size,
                max_size=max_size,
                kwargs={"row_factory": dict_row},
                open=True,
            )
        self._init_pgvector_schema()

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()

    def _get_connection(self):
        if self._pool is not None:
            return self._pool.connection()
        return psycopg.connect(self.dsn, row_factory=dict_row)

    def _init_pgvector_schema(self) -> None:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS chunk_vectors (
                        chunk_id TEXT PRIMARY KEY REFERENCES document_chunks(chunk_id) ON DELETE CASCADE,
                        doc_id TEXT NOT NULL,
                        file_id TEXT NOT NULL REFERENCES raw_files(file_id) ON DELETE CASCADE,
                        chunk_index INTEGER NOT NULL,
                        content_type TEXT NOT NULL,
                        text TEXT NOT NULL,
                        metadata JSONB NOT NULL,
                        embedding vector({self.dimension}) NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chunk_vectors_hnsw
                    ON chunk_vectors USING hnsw (embedding vector_cosine_ops);
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS document_index_status (
                        file_id TEXT PRIMARY KEY REFERENCES raw_files(file_id) ON DELETE CASCADE,
                        status TEXT NOT NULL,
                        indexed_chunks INTEGER NOT NULL DEFAULT 0,
                        error TEXT,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                    );
                """)
            conn.commit()

    def upsert_chunks(self, chunks: List[DocumentChunk]) -> int:
        if not chunks:
            return 0
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                for c in chunks:
                    if not c.embedding:
                        raise ValueError(f"Chunk '{c.chunk_id}' has no embedding vector.")
                    vec_literal = "[" + ",".join(f"{val:.6f}" for val in c.embedding) + "]"
                    meta_dict = {
                        "source_name": c.metadata.source_name,
                        "version_date": c.metadata.version_date,
                        "source_link": c.metadata.source_link,
                        "file_format": c.metadata.file_format,
                        "page_or_section": c.metadata.page_or_section,
                        "extra": c.metadata.extra,
                    }
                    cur.execute(
                        """
                        INSERT INTO chunk_vectors
                        (chunk_id, doc_id, file_id, chunk_index, content_type, text, metadata, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
                        ON CONFLICT (chunk_id) DO UPDATE
                        SET embedding = EXCLUDED.embedding,
                            text = EXCLUDED.text,
                            metadata = EXCLUDED.metadata;
                        """,
                        (
                            c.chunk_id,
                            c.doc_id,
                            c.file_id,
                            c.chunk_index,
                            c.content_type,
                            c.text,
                            Jsonb(meta_dict),
                            vec_literal,
                        ),
                    )
            conn.commit()
        return len(chunks)

    def search(
        self,
        query_embedding: List[float],
        query_text: str,
        top_k: int = 5,
        mode: str = "hybrid",
        file_id_filter: Optional[str] = None,
        content_type_filter: Optional[str] = None,
    ) -> List[DocumentChunk]:
        vec_literal = "[" + ",".join(f"{val:.6f}" for val in query_embedding) + "]"
        where_clauses = []
        params: List[object] = [vec_literal]

        if file_id_filter:
            where_clauses.append("file_id = %s")
            params.append(file_id_filter)
        if content_type_filter:
            where_clauses.append("content_type = %s")
            params.append(content_type_filter)

        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        candidate_limit = max(top_k * 4, 25)
        params.append(vec_literal)
        params.append(candidate_limit)

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT chunk_id, doc_id, file_id, chunk_index, content_type, text, metadata,
                           1 - (embedding <=> %s::vector) AS cosine_sim
                    FROM chunk_vectors
                    {where_sql}
                    ORDER BY embedding <=> %s::vector ASC
                    LIMIT %s;
                    """,
                    tuple(params),
                )
                rows = cur.fetchall()

        if not rows:
            return []

        candidates: List[DocumentChunk] = []
        sem_scores: Dict[str, float] = {}
        lex_scores: Dict[str, float] = {}

        for r in rows:
            m = r["metadata"]
            meta = SourceMetadata(
                source_name=m["source_name"],
                version_date=m["version_date"],
                source_link=m["source_link"],
                file_format=m["file_format"],
                page_or_section=m.get("page_or_section"),
                extra=m.get("extra", {}),
            )
            chunk = DocumentChunk(
                chunk_id=r["chunk_id"],
                doc_id=r["doc_id"],
                file_id=r["file_id"],
                chunk_index=r["chunk_index"],
                content_type=r["content_type"],
                text=r["text"],
                metadata=meta,
            )
            candidates.append(chunk)
            sem_scores[chunk.chunk_id] = float(r["cosine_sim"])
            lex_scores[chunk.chunk_id] = _lexical_overlap_score(query_text, chunk.text)

        fused = _apply_rrf_fusion(candidates, sem_scores, lex_scores, mode=mode, rrf_k=self.rrf_k)
        return fused[:top_k]

    def delete_by_file_id(self, file_id: str) -> int:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chunk_vectors WHERE file_id = %s;", (file_id,))
                deleted = cur.rowcount
            conn.commit()
        return deleted

    def count_vectors(self, file_id: Optional[str] = None) -> int:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                if file_id:
                    cur.execute("SELECT COUNT(*) AS cnt FROM chunk_vectors WHERE file_id = %s;", (file_id,))
                else:
                    cur.execute("SELECT COUNT(*) AS cnt FROM chunk_vectors;")
                return int(cur.fetchone()["cnt"])

    def set_index_status(self, file_id: str, status: str, indexed_chunks: int = 0, error: Optional[str] = None) -> None:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO document_index_status (file_id, status, indexed_chunks, error, updated_at)
                       VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                       ON CONFLICT (file_id) DO UPDATE SET status = EXCLUDED.status,
                       indexed_chunks = EXCLUDED.indexed_chunks, error = EXCLUDED.error,
                       updated_at = CURRENT_TIMESTAMP""",
                    (file_id, status, indexed_chunks, error),
                )
            conn.commit()

    def get_index_statuses(self) -> Dict[str, Dict[str, object]]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT file_id, status, indexed_chunks, error, updated_at FROM document_index_status")
                rows = cur.fetchall()
        return {
            row["file_id"]: {
                "status": row["status"], "indexed_chunks": row["indexed_chunks"],
                "error": row["error"], "updated_at": row["updated_at"].isoformat(),
            }
            for row in rows
        }
