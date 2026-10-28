from pathlib import Path


def prepare_sqlite_dsn(dsn: str) -> str:
    """Create the parent directory for a file-backed SQLite database."""
    if not dsn or dsn == ":memory:" or dsn.startswith("file:"):
        return dsn

    database_path = Path(dsn).expanduser()
    parent = database_path.parent
    if parent != Path("."):
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(
                "SQLite database directory is unavailable. "
                "Set RAG_DB_DSN to a writable persistent path."
            ) from exc
    return str(database_path)
