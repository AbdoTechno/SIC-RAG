# Pulmora mobile integration

## Connection and authentication

- Production base URL: `https://sic-rag.bot-keep.xyz`
- OpenAPI: `GET /openapi.json`
- Public checks: `GET /` and `GET /health`
- All data, retrieval, generation, and readiness routes require `X-API-Key` when `RAG_API_KEY` is configured.
- Send an optional `X-Request-ID`; the service returns it in the response header and typed error body.

The shared API key is acceptable only for an internal prototype. A key inside a public mobile binary can be recovered and is not user authorization. Do not ship `GROQ_API_KEY` or `RAG_DB_DSN` to Flutter.

## Readiness

Before enabling case review, call authenticated `GET /knowledge/status`. `retrieval_ready` is true only when at least one reference exists and every registered document has exactly as many persisted vectors as chunks. This endpoint reports document-level `index_status`, counts, embedding/index configuration, and generation configuration availability without calling the LLM provider.

`GET /health` only confirms service/database health; it does not prove that evidence retrieval is ready.

## Generic reference question

`POST /rag/generate`

```json
{
  "query": "How does the reference define hospital-acquired pneumonia?",
  "top_k": 3,
  "mode": "hybrid",
  "file_id_filter": null,
  "abstention_threshold_override": null
}
```

The backward-compatible response contains `query`, `abstained`, `model_name`, `answer`, `token_usage`, and `sources_cited`.

## Assessment case review

`POST /rag/case-review`

The complete synthetic request is in `examples/case_review_request.json`. Required correlation fields must be generated and retained by Flutter:

```json
{
  "request_id": "flutter-2026-10-28-0001",
  "assessment_id": "assessment-opaque-7d55",
  "input_revision": 1,
  "question": "Which reference evidence is relevant to this assessment?",
  "clinical_context": {
    "age_years": 58,
    "sex": "male",
    "chief_complaint": "Shortness of breath",
    "symptoms": [{"name": "dyspnea", "duration": "3 days"}],
    "chronic_conditions": {"status": "unknown", "items": []},
    "allergies": {"status": "none", "items": []},
    "medications": {"status": "unknown", "items": []},
    "vitals": [{"name": "oxygen_saturation", "value": 91, "unit": "%", "recorded_at": null}],
    "other_tests": [],
    "doctor_notes": ""
  },
  "imaging_findings": {
    "origin": "local_model",
    "model_name": "pulmora-local-cxr",
    "model_version": "1",
    "findings": [{"label": "pleural effusion", "score": 0.82, "score_kind": "confidence", "calibrated": false}],
    "warnings": []
  },
  "top_k": 3,
  "mode": "hybrid",
  "file_id_filter": null
}
```

An empty `findings` array is valid. Never upload the raw X-ray to this endpoint. A local-model label is treated as an unconfirmed finding; an uncalibrated score is not interpreted as disease probability.

Successful response shape:

```json
{
  "request_id": "flutter-2026-10-28-0001",
  "assessment_id": "assessment-opaque-7d55",
  "input_revision": 1,
  "question": "Which reference evidence is relevant to this assessment?",
  "abstained": false,
  "abstention_reason": null,
  "model_name": "configured-model",
  "answer": "Same supported content as evidence_discussion.",
  "case_summary": "Summary of only the submitted snapshot.",
  "evidence_discussion": "Grounded discussion with [Source 1] references.",
  "missing_information": [],
  "review_points": [],
  "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
  "sources_cited": [
    {
      "source_ref": "[Source 1]",
      "cited_in_answer": true,
      "chunk_id": "...",
      "file_id": "...",
      "source_name": "...",
      "section": "Page 12 | ...",
      "pdf_page": 12,
      "snippet": "...",
      "rerank_score": 0.73,
      "organization": "...",
      "publication_date": "2023",
      "source_url": "https://..."
    }
  ],
  "service": {"name": "rag-engine", "version": "3.1.0"}
}
```

`answer` is retained for generic UI compatibility and equals `evidence_discussion`. `case_summary` is based on submitted data and is not guideline evidence. `sources_cited` contains the retrieved evidence set; use `cited_in_answer` to identify inline citations. `pdf_page` is the 1-based physical PDF position, not a printed page label. Retrieval/rerank scores measure matching, not medical confidence.

If evidence is insufficient, HTTP remains 200 with `abstained: true`, a safe `abstention_reason`, empty evidence fields, and an empty source list.

## Flutter mapping

- Build each request from the current Hive assessment snapshot; the server is stateless and does not own visit history.
- Map the current assessment's opaque ID and revision without patient name, phone, DOB, or image bytes.
- Map symptom names/durations, known conditions, allergies, medications, vitals with their original units, other test text, and doctor notes.
- Use `unknown` when not recorded and `none` only when explicitly confirmed absent. `items` must be empty for both states; use `known` with nonempty/empty items only as clinically recorded.
- Map the already-integrated local CXR output to `imaging_findings`; preserve label, score, score kind, calibration flag, and model version.
- Discard stale responses when `assessment_id` or `input_revision` no longer matches the visible assessment.
- Render citations by joining `[Source N]` to `source_ref`; offer the source title, section/page, snippet, and URL.
- Keep a valid abstention as a normal result, not a transport error.

## Typed errors

Errors contain `detail` for backward compatibility plus:

```json
{
  "error": {
    "code": "INDEX_NOT_READY",
    "message": "Reference documents are not fully indexed.",
    "request_id": "...",
    "retryable": true
  }
}
```

Stable codes include `UNAUTHORIZED` (401), `VALIDATION_ERROR` (422), `PAYLOAD_TOO_LARGE` (413), `KNOWLEDGE_BASE_EMPTY` (400), `INDEX_NOT_READY` (400), `GENERATION_UNAVAILABLE` (503), and `INVALID_MODEL_RESPONSE` (502). Retry only when `retryable` is true and use bounded backoff.

## Persistence and deployment

SQLite document/chunk/vector/index-status tables share the configured database and survive an application restart when the database file itself is on persistent storage. Container-local files must not be assumed to survive a BotKeep redeploy. For deployment, either mount a BotKeep persistent volume and set `RAG_STORE_PROVIDER=sqlite_store` plus an absolute volume path in `RAG_DB_DSN`, or use PostgreSQL with `pgvector` using the existing provider configuration. Confirm the hosting storage guarantee before production ingestion.

After deploying a revision, run the corpus ingestion command once against the deployed durable store, then verify `/knowledge/status`. Ingestion is deliberately not tied to server startup.
