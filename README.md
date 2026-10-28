# ✦ Modular RAG Engine

> A domain-agnostic, schema-driven End-to-End Retrieval-Augmented Generation (RAG) architecture engineered on strict SOLID principles, ACID persistence, and deterministic execution.

![Python Version](https://img.shields.io/badge/python-3.11%2B-1f2937.svg?style=flat-square&logo=python&logoColor=white)
![Architecture](https://img.shields.io/badge/architecture-SOLID%20%7C%20Plug--and--Play-0f172a.svg?style=flat-square)
![Data Integrity](https://img.shields.io/badge/persistence-ACID%20%7C%20Idempotent-111827.svg?style=flat-square)

The **Modular RAG Engine** is a full-lifecycle document intelligence, retrieval, and grounded generation framework. Rather than coupling system logic to specific databases, file systems, or model providers, the engine treats every stage of the RAG lifecycle—parsing, normalization, chunking, relational persistence, vector indexing, reranking, context augmentation, and LLM synthesis—as an interchangeable, contract-bound plugin orchestrated via declarative configurations.

---

## Engineering Philosophy

- **Declarative Configuration & Registry Binding:** Runtime behavior is governed by `pipeline_config.yaml` and `prompts.yaml`. A Singleton Registry resolves provider keys into concrete class implementations without hard-coded dependencies.
- **Zero-Filesystem Byte-Stream Ingestion:** Raw files are processed directly from memory streams into ACID relational storage (`BYTEA`/`BLOB`), enabling stateless container execution.
- **Two-Tier Cryptographic Idempotency:**
  - *Payload Deduplication:* `SHA-256` binary hashing prevents duplicate storage of identical files and dynamically maps HTTP responses (`201 Created` vs. `200 OK`).
  - *Execution Caching:* Configuration fingerprinting (`config_hash`) caches chunking and indexing states per parameter profile, bypassing redundant compute unless explicitly overridden.
- **Unbroken Evidence Provenance:** Immutable value objects (`SourceMetadata`) propagate document origin, version timestamps, and section headers from raw ingestion through to the final LLM citation payload.
- **Selective Abstention Gate:** Retrieval scores are evaluated against strict confidence thresholds prior to LLM invocation, preventing hallucinated synthesis when supporting evidence is insufficient.

---

## End-to-End System Architecture

The diagram below depicts the complete system topology across three planes: **Control & Configuration**, **Write Path (Ingestion & Indexing)**, and **Read Path (Retrieval, Augmentation & Generation)**. 
*(Legend: `●` Implemented | `◌` Active Integration)*


<img alt="FastAPI Qdrant Retrieval Evaluation Pipeline-2026-10-25-060312" src="https://github.com/user-attachments/assets/4d35b856-9ca4-4117-b0f4-6a2678a6ee32" />

---

## Run locally

Python 3.11 is recommended.

```bash
python -m venv .venv
# Activate the environment, then:
pip install -r requirements.txt
docker compose up -d
python start.py
```

The API is available at `http://127.0.0.1:8000`, interactive documentation at
`/docs`, and health status at `/health`.

Copy `.env.example` to `.env` only for local development. Never commit real
credentials. The production database must be PostgreSQL with the `pgvector`
extension enabled.

## Deploy to BotKeep

1. Push this prepared revision to a GitHub repository you control.
2. In BotKeep, create a Python application from that repository and select the
   repository root (the directory containing `requirements.txt`).
3. Select Python 3.11 and use this start command:

   ```text
   python start.py
   ```

4. Add the following in BotKeep Environment/Secrets:

   - `RAG_DB_DSN`: PostgreSQL connection URL for a database with `pgvector`.
   - `GROQ_API_KEY`: Groq secret key used for answer generation.
   - `RAG_API_KEY`: a long random shared key required in `X-API-Key`.
   - `RAG_CORS_ORIGINS`: comma-separated Flutter Web origins. It is not needed
     for Android/iOS clients.
   - `RAG_DB_POOL_MAX=3`: conservative connection limit for a small instance.
   - `RAG_MAX_UPLOAD_MB=20`: maximum accepted document size.

   BotKeep supplies `SERVER_PORT`; do not hard-code it in the panel or source.

5. Enable HTTPS routing in **Domains**, then verify:

   ```text
   GET https://YOUR_DOMAIN/health
   GET https://YOUR_DOMAIN/docs
   ```

BotKeep databases consume a separate workload slot. If the account has only one
slot, use an external PostgreSQL provider that supports `pgvector`. Do not use
the SQLite/in-memory testing mode for production: the vector index would not be
durable across restarts.

## API contract

All non-public routes require the `X-API-Key` header when `RAG_API_KEY` is set.
Public routes are `/`, `/health`, `/docs`, `/redoc`, and `/openapi.json`.

Typical application flow:

1. `POST /documents` as `multipart/form-data` with `file`, `source_name`, and
   optional `version_date`, `source_link`, and `auto_chunk`.
2. `POST /documents/{file_id}/index` after upload/chunking.
3. `POST /rag/generate` with JSON such as:

   ```json
   {
     "query": "What does the uploaded guideline say?",
     "top_k": 3,
     "mode": "hybrid",
     "file_id_filter": null
   }
   ```

4. Display `answer`, `abstained`, and `sources_cited` from the response.

Other useful routes include `GET /documents`, `GET /documents/{file_id}`,
`DELETE /documents/{file_id}`, `GET /documents/{file_id}/chunks`,
`POST /retrieval/search`, and `GET /rag/models`. The generated OpenAPI contract
at `/openapi.json` is the source of truth for request and response schemas.

## Production notes

- Keep `GROQ_API_KEY`, `RAG_DB_DSN`, and `RAG_API_KEY` only in BotKeep Secrets.
- `RAG_API_KEY` is basic shared-secret protection. A key embedded in a mobile
  binary can be extracted; use user authentication or a trusted backend proxy
  before handling private or per-user data.
- The service intentionally runs one Uvicorn process because the in-process
  pipelines and small BotKeep resource profile do not benefit from extra workers.
- Monitor `/health`, application logs, database connection usage, and Groq quota.

## Tests

```bash
uv sync --dev
uv run pytest -q
```

