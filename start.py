"""BotKeep-compatible production entry point."""

import os

import uvicorn


def _port() -> int:
    raw_port = os.getenv("SERVER_PORT") or os.getenv("PORT") or "8000"
    port = int(raw_port)
    if not 1 <= port <= 65535:
        raise ValueError("SERVER_PORT/PORT must be between 1 and 65535.")
    return port


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        app_dir="rag_engine",
        host="0.0.0.0",
        port=_port(),
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
