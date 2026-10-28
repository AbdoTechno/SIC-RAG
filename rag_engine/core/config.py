import os
from pathlib import Path
from typing import Any, Dict

import yaml


def load_default_config() -> Dict[str, Any]:
    config_path = Path(__file__).resolve().parent.parent / "configs" / "pipeline_config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    env_provider = os.getenv("RAG_STORE_PROVIDER")
    env_dsn = os.getenv("RAG_DB_DSN")
    if env_provider:
        cfg["document_store"]["provider"] = env_provider
        if env_provider == "sqlite_store":
            sqlite_dsn = env_dsn or ":memory:"
            cfg["document_store"]["params"] = {"dsn": sqlite_dsn}
            cfg.setdefault("retrieval", {})["vector_store"] = {
                "provider": "sqlite_vector_store",
                "params": {"dsn": sqlite_dsn},
            }
    if env_dsn and env_provider != "sqlite_store":
        cfg["document_store"]["params"]["dsn"] = env_dsn
        if cfg.get("retrieval", {}).get("vector_store", {}).get("provider") == "pgvector_store":
            cfg["retrieval"]["vector_store"]["params"]["dsn"] = env_dsn

    env_pool_max = os.getenv("RAG_DB_POOL_MAX")
    if env_pool_max and cfg["document_store"]["provider"] == "postgres_store":
        pool_max = int(env_pool_max)
        if pool_max < 1:
            raise ValueError("RAG_DB_POOL_MAX must be at least 1.")
        cfg["document_store"]["params"]["max_size"] = pool_max
        vector_params = cfg.get("retrieval", {}).get("vector_store", {}).get("params", {})
        if vector_params:
            vector_params["max_size"] = pool_max

    gen_params = cfg.setdefault("generation", {}).setdefault("params", {})
    if os.getenv("RAG_LLM_MODEL"):
        gen_params["model_name"] = os.environ["RAG_LLM_MODEL"]
    if os.getenv("RAG_LLM_TEMP"):
        gen_params["temperature"] = float(os.environ["RAG_LLM_TEMP"])
    if os.getenv("RAG_LLM_MAX_TOKENS"):
        gen_params["max_tokens"] = int(os.environ["RAG_LLM_MAX_TOKENS"])

    if os.getenv("RAG_ENFORCE_SCOPE"):
        cfg.setdefault("retrieval", {})["enforce_scope"] = os.environ["RAG_ENFORCE_SCOPE"].lower() in {"1", "true", "yes"}

    return cfg
