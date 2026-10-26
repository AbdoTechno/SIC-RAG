import os
from typing import List, Dict, Any, Optional
from core.interfaces import BaseGenerator
from core.registry import registry


@registry.register_generator("groq_generator")
class GroqGenerator(BaseGenerator):
    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._explicit_api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            resolved_key = self._explicit_api_key or os.getenv("GROQ_API_KEY")
            if not resolved_key:
                raise ValueError("Groq API key is missing. Set the GROQ_API_KEY environment variable.")
            from groq import Groq
            self._client = Groq(api_key=resolved_key)
        return self._client

    def generate(self, query: str, context_chunks: List[Any]) -> Dict[str, Any]:
        client = self._get_client()
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
            model=self.model_name,
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
            "model_name": self.model_name,
            "token_usage": usage,
        }


@registry.register_generator("extractive_mock_generator")
class ExtractiveMockGenerator(BaseGenerator):
    def __init__(self, model_name: str = "extractive-deterministic-v1"):
        self.model_name = model_name

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
            "token_usage": {"prompt_tokens": len(query.split()), "completion_tokens": len(synthesized.split()), "total_tokens": len(query.split()) + len(synthesized.split())},
        }
