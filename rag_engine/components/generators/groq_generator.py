import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from core.interfaces import BaseGenerator
from core.registry import registry


def _load_dotenv_override() -> None:
    env_path = Path(__file__).resolve().parent.parent.parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip('"').strip("'")


@registry.register_generator("groq_generator")
class GroqGenerator(BaseGenerator):
    EXCLUDED_KEYWORDS = ("whisper", "orpheus", "tts", "guard", "playai", "embed")
    PREFERRED_DEFAULTS = ("allam-2-7b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b", "openai/gpt-oss-120b")

    def __init__(
        self,
        model_name: str = "allam-2-7b",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        api_key: Optional[str] = None,
    ):
        _load_dotenv_override()
        self.model_name = os.getenv("RAG_LLM_MODEL") or model_name
        self.temperature = float(os.getenv("RAG_LLM_TEMP", temperature))
        self.max_tokens = int(os.getenv("RAG_LLM_MAX_TOKENS", max_tokens))
        self._explicit_api_key = api_key
        self._client = None
        self._available_models: Optional[List[str]] = None

    def _get_client(self):
        if self._client is None:
            _load_dotenv_override()
            resolved_key = self._explicit_api_key or os.getenv("GROQ_API_KEY")
            if not resolved_key:
                raise ValueError("Groq API key is missing. Set GROQ_API_KEY in .env or environment.")
            from groq import Groq
            self._client = Groq(api_key=resolved_key)
        return self._client

    def list_available_models(self) -> List[str]:
        if self._available_models is None:
            client = self._get_client()
            raw_models = [m.id for m in client.models.list().data]
            self._available_models = sorted([
                m for m in raw_models
                if not any(ex in m.lower() for ex in self.EXCLUDED_KEYWORDS)
            ])
        return self._available_models

    def _resolve_active_model(self) -> str:
        available = self.list_available_models()
        if not available:
            return self.model_name
        if self.model_name in available:
            return self.model_name
        for pref in self.PREFERRED_DEFAULTS:
            if pref in available:
                self.model_name = pref
                return self.model_name
        self.model_name = available[0]
        return self.model_name

    def generate(self, query: str, context_chunks: List[Any]) -> Dict[str, Any]:
        client = self._get_client()
        active_model = self._resolve_active_model()

        context_text = "\n\n".join(
            f"[Source {i+1}] (Section: {c.metadata.page_or_section} | Source: {c.metadata.source_name})\n{c.text}"
            for i, c in enumerate(context_chunks)
        )

        system_prompt = (
            "You are a precise, evidence-grounded AI assistant. Answer the user's question strictly using ONLY the provided context chunks. "
            "Every factual claim MUST include an inline citation in the format [Source X] corresponding to the source block. "
            "Do not extrapolate or invent facts outside the context. If the provided chunks do not answer the question, state clearly that the evidence is insufficient."
        )

        user_prompt = f"Context Chunks:\n{context_text}\n\nQuestion: {query}\n\nEvidence-Linked Answer:"

        response = client.chat.completions.create(
            model=active_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        answer = response.choices[0].message.content or ""
        usage = {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
            "completion_tokens": getattr(response.usage, "completion_tokens", 0),
            "total_tokens": getattr(response.usage, "total_tokens", 0),
        }

        return {
            "answer": answer.strip(),
            "model_name": active_model,
            "token_usage": usage,
        }


@registry.register_generator("extractive_mock_generator")
class ExtractiveMockGenerator(BaseGenerator):
    def __init__(self, model_name: str = "extractive-deterministic-v1"):
        self.model_name = model_name

    def list_available_models(self) -> List[str]:
        return [self.model_name]

    def generate(self, query: str, context_chunks: List[Any]) -> Dict[str, Any]:
        if not context_chunks:
            return {
                "answer": "Insufficient context to answer.",
                "model_name": self.model_name,
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        top = context_chunks[0]
        synthesized = f"Based on {top.metadata.page_or_section}: {top.text} [Source 1]"
        return {
            "answer": synthesized,
            "model_name": self.model_name,
            "token_usage": {
                "prompt_tokens": len(query.split()),
                "completion_tokens": len(synthesized.split()),
                "total_tokens": len(query.split()) + len(synthesized.split()),
            },
        }
