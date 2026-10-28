import gzip
import hashlib
import sqlite3
from pathlib import Path

from rag_engine.core.seed_bootstrap import bootstrap_sqlite_seed


def _seed_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE raw_files (file_id TEXT);
            CREATE TABLE document_chunks (chunk_id TEXT);
            CREATE TABLE chunk_vectors (chunk_id TEXT);
            INSERT INTO raw_files VALUES ('file-1');
            INSERT INTO document_chunks VALUES ('chunk-1');
            INSERT INTO chunk_vectors VALUES ('chunk-1');
            """
        )
        connection.commit()
    finally:
        connection.close()


def test_bootstrap_joins_split_seed_parts(tmp_path, monkeypatch):
    source = tmp_path / "source.db"
    seed = tmp_path / "seed.db.gz"
    target = tmp_path / "runtime" / "rag.db"
    _seed_database(source)
    with source.open("rb") as raw, gzip.open(seed, "wb") as compressed:
        compressed.write(raw.read())
    payload = seed.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    Path(f"{seed}.sha256").write_text(f"{digest}  {seed.name}\n", encoding="utf-8")
    midpoint = len(payload) // 2
    Path(f"{seed}.part-001").write_bytes(payload[:midpoint])
    Path(f"{seed}.part-002").write_bytes(payload[midpoint:])
    seed.unlink()

    monkeypatch.setenv("RAG_STORE_PROVIDER", "sqlite_store")
    monkeypatch.setenv("RAG_BOOTSTRAP_SEED", "true")
    assert bootstrap_sqlite_seed(str(target), seed) is True


def test_bootstrap_installs_seed_only_into_empty_sqlite_database(tmp_path, monkeypatch):
    source = tmp_path / "source.db"
    seed = tmp_path / "seed.db.gz"
    target = tmp_path / "runtime" / "rag.db"
    _seed_database(source)
    with source.open("rb") as raw, gzip.open(seed, "wb") as compressed:
        compressed.write(raw.read())
    digest = hashlib.sha256(seed.read_bytes()).hexdigest()
    Path(f"{seed}.sha256").write_text(f"{digest}  {seed.name}\n", encoding="utf-8")

    monkeypatch.setenv("RAG_STORE_PROVIDER", "sqlite_store")
    monkeypatch.setenv("RAG_BOOTSTRAP_SEED", "true")
    assert bootstrap_sqlite_seed(str(target), seed) is True
    assert bootstrap_sqlite_seed(str(target), seed) is False

    connection = sqlite3.connect(target)
    try:
        assert connection.execute("SELECT COUNT(*) FROM raw_files").fetchone()[0] == 1
    finally:
        connection.close()
