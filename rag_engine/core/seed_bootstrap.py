import gzip
import hashlib
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Optional


def _enabled(value: Optional[str]) -> bool:
    return (value or "true").strip().lower() in {"1", "true", "yes", "on"}


def _has_ready_knowledge(database_path: Path) -> bool:
    if not database_path.is_file() or database_path.stat().st_size == 0:
        return False
    try:
        connection = sqlite3.connect(
            f"file:{database_path.as_posix()}?mode=ro",
            uri=True,
        )
        try:
            raw_files = int(connection.execute("SELECT COUNT(*) FROM raw_files").fetchone()[0])
            chunks = int(connection.execute("SELECT COUNT(*) FROM document_chunks").fetchone()[0])
            vectors = int(connection.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0])
            return raw_files > 0 and chunks > 0 and chunks == vectors
        finally:
            connection.close()
    except (OSError, sqlite3.Error):
        return False


def _verify_seed(seed_path: Path) -> None:
    checksum_path = Path(f"{seed_path}.sha256")
    if not checksum_path.is_file():
        raise RuntimeError("Bundled knowledge seed checksum is missing.")
    expected = checksum_path.read_text(encoding="utf-8").strip().split()[0].lower()
    digest = hashlib.sha256()
    with seed_path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise RuntimeError("Bundled knowledge seed checksum validation failed.")


def bootstrap_sqlite_seed(
    target_dsn: Optional[str] = None,
    seed_path: Optional[Path] = None,
) -> bool:
    """Install the bundled corpus into an empty SQLite deployment."""
    if os.getenv("RAG_STORE_PROVIDER") != "sqlite_store":
        return False
    if not _enabled(os.getenv("RAG_BOOTSTRAP_SEED")):
        return False

    resolved_dsn = target_dsn or os.getenv("RAG_DB_DSN", "")
    if not resolved_dsn or resolved_dsn == ":memory:" or resolved_dsn.startswith("file:"):
        return False

    target_path = Path(resolved_dsn).expanduser()
    if _has_ready_knowledge(target_path):
        return False

    resolved_seed = seed_path or Path(
        os.getenv("RAG_SQLITE_SEED_PATH", "seed/pulmora_rag.db.gz")
    )
    if not resolved_seed.is_file():
        return False

    _verify_seed(resolved_seed)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="pulmora-seed-",
            suffix=".db",
            dir=target_path.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            with gzip.open(resolved_seed, "rb") as source:
                shutil.copyfileobj(source, temporary)

        temporary_path = Path(temporary_name)
        if not _has_ready_knowledge(temporary_path):
            raise RuntimeError("Bundled knowledge seed is incomplete.")
        os.replace(temporary_path, target_path)
        return True
    finally:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)
