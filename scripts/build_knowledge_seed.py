"""Build a validated, compressed SQLite knowledge seed for deployment."""

import argparse
import gzip
import hashlib
import shutil
import sqlite3
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=Path("data/pulmora_rag.db"))
    parser.add_argument("--output", type=Path, default=Path("seed/pulmora_rag.db.gz"))
    args = parser.parse_args()

    if not args.db.is_file():
        raise SystemExit("Ingested SQLite database is missing.")

    connection = sqlite3.connect(f"file:{args.db.as_posix()}?mode=ro", uri=True)
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        documents = int(connection.execute("SELECT COUNT(*) FROM raw_files").fetchone()[0])
        chunks = int(connection.execute("SELECT COUNT(*) FROM document_chunks").fetchone()[0])
        vectors = int(connection.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0])
    finally:
        connection.close()
    if integrity != "ok" or documents == 0 or chunks == 0 or chunks != vectors:
        raise SystemExit("SQLite knowledge database is not ready for packaging.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{args.output}.tmp")
    with args.db.open("rb") as source, gzip.open(temporary, "wb", compresslevel=9) as target:
        shutil.copyfileobj(source, target)
    temporary.replace(args.output)

    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    Path(f"{args.output}.sha256").write_text(f"{digest}  {args.output.name}\n", encoding="utf-8")
    print(f"Packaged {documents} documents, {chunks} chunks, and {vectors} vectors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
