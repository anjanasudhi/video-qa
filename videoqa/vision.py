"""AI review of the rules that need judgement, using a video-capable LLM."""

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Protocol

from .formats import gemini_ready
from .llm import DEFAULT_GEMINI_MODEL, call_with_retries
from .models import Rule
from .probe import VideoInfo

SYSTEM_PROMPT = """You are a meticulous brand compliance reviewer. You watch a video and decide, rule by rule, \
whether it complies with a brand guideline. Judge only what you can actually see or hear; if the evidence is \
ambiguous, say so in the explanation and lower the confidence. Never assume a rule is met because the video \
looks professional."""


class VideoReviewer(Protocol):
    def review(self, video: Path, info: VideoInfo, rules: list[Rule]) -> list[dict[str, Any]]: ...


def build_review_prompt(info: VideoInfo, rules: list[Rule]) -> str:
    checklist = "\n".join(f'{i}. [{r.id}] {r.criterion}' for i, r in enumerate(rules, start=1))
    return f"""Review the attached video. Known facts: {info.width}x{info.height} px, {info.fps:.0f} fps, {info.duration:.1f} seconds.

For EACH question below, answer whether the video complies ("yes" = complies).
{checklist}

Return a JSON array with exactly one object per question, in the same order:
  rule_id      the id in square brackets
  complies     true or false
  confidence   number from 0 to 1
  explanation  one or two sentences describing what you saw or heard
  timestamps   seconds (numbers) where a violation is visible or audible; empty when it complies
"""


class GeminiVideoReviewer:
    """Uploads the clip to the Gemini Files API and reviews all AI rules in one request."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        from google import genai

        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("No Gemini API key found. Set GEMINI_API_KEY (free key: https://aistudio.google.com/apikey).")
        self.client = genai.Client(api_key=key)
        self.model = model or os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)

    def review(self, video: Path, info: VideoInfo, rules: list[Rule]) -> list[dict[str, Any]]:
        with tempfile.TemporaryDirectory() as tmp:
            return self._review(gemini_ready(Path(video), Path(tmp)), info, rules)

    def _review(self, upload: Path, info: VideoInfo, rules: list[Rule]) -> list[dict[str, Any]]:
        from google.genai import types

        uploaded = self.client.files.upload(file=str(upload))
        try:
            while uploaded.state and uploaded.state.name == "PROCESSING":
                time.sleep(2)
                uploaded = self.client.files.get(name=uploaded.name)
            if uploaded.state and uploaded.state.name == "FAILED":
                raise RuntimeError("Gemini could not process the video file")

            def call():
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[uploaded, build_review_prompt(info, rules)],
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT, response_mime_type="application/json", temperature=0.0,
                    ),
                )
                return json.loads(response.text)

            return call_with_retries(call)
        finally:
            try:
                self.client.files.delete(name=uploaded.name)
            except Exception:
                pass


class MockReviewer:
    """Canned verdicts keyed by rule id; for tests and offline development."""

    def __init__(self, verdicts: dict[str, dict[str, Any]]):
        self.verdicts = verdicts
        self.calls: list[list[str]] = []

    def review(self, video: Path, info: VideoInfo, rules: list[Rule]) -> list[dict[str, Any]]:
        self.calls.append([r.id for r in rules])
        return [{"rule_id": r.id, **self.verdicts[r.id]} for r in rules if r.id in self.verdicts]
