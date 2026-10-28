# Bundled knowledge seed

`pulmora_rag.db.gz` is generated from the reviewed local corpus with:

```bash
python -m scripts.build_knowledge_seed --db data/pulmora_rag.db --output seed/pulmora_rag.db.gz
```

For a SQLite deployment, `start.py` installs this seed atomically when the configured database is missing, empty, corrupt, or does not contain a complete chunk/vector index. It never replaces a database that already contains ready knowledge.

Set `RAG_BOOTSTRAP_SEED=false` to disable automatic installation, or `RAG_SQLITE_SEED_PATH` to use a different seed artifact.

The seed contains processed reference content. Review the source publishers' licensing and redistribution terms before using it outside the authorized deployment.
