"""LLM backends. Anything with generate_json(system, prompt) -> parsed JSON can be swapped in."""

import json
import os
import time
from pathlib import Path
from typing import Any, Protocol

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"


class LLM(Protocol):
    def generate_json(self, system: str, prompt: str) -> Any: ...


def call_with_retries(fn, attempts: int = 4):
    """Gemini's free tier returns transient 5xx under load and occasionally malformed JSON; retry both."""
    from google.genai import errors

    for attempt in range(attempts):
        try:
            return fn()
        except (errors.ServerError, json.JSONDecodeError):
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** (attempt + 1))


class GeminiLLM:
    """Gemini via the Google AI Studio API; works on the free tier with a GEMINI_API_KEY."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        from google import genai

        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "No Gemini API key found. Create a free key at https://aistudio.google.com/apikey "
                "and set GEMINI_API_KEY."
            )
        self.client = genai.Client(api_key=key)
        self.model = model or os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)

    def generate_json(self, system: str, prompt: str) -> Any:
        from google.genai import types

        def call():
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                    temperature=0.1,
                    thinking_config=types.ThinkingConfig(thinking_level="low"),
                ),
            )
            return json.loads(response.text)

        return call_with_retries(call)


class MockLLM:
    """Returns a canned response; used for offline runs and tests."""

    def __init__(self, response: Any = None, fixture: str | Path | None = None):
        if fixture is not None:
            response = json.loads(Path(fixture).read_text())
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def generate_json(self, system: str, prompt: str) -> Any:
        self.calls.append((system, prompt))
        return self.response


def get_llm(backend: str = "gemini", **kwargs) -> LLM:
    if backend == "gemini":
        return GeminiLLM(**kwargs)
    if backend == "mock":
        return MockLLM(**kwargs)
    raise ValueError(f"Unknown LLM backend: {backend}")
