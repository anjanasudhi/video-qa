"""Turn a brand guideline PDF into a list of structured, checkable video rules."""

import json
import re
from pathlib import Path

from pypdf import PdfReader

from .llm import LLM
from .models import TECHNICAL_CHECKS, AiScope, Brand, CheckType, Rule

SYSTEM_PROMPT = """You are a brand compliance analyst. You read a brand guideline and extract every rule \
that can be checked on a finished video asset (visuals, on-screen text, audio, technical specs, structure). \
Ignore rules that only concern print, web layout, photography or anything not observable in a video, \
unless the guideline says they also apply to video."""


def read_pdf_text(path: str | Path) -> str:
    reader = PdfReader(str(path))
    pages = [f"--- Page {i} ---\n{page.extract_text() or ''}" for i, page in enumerate(reader.pages, start=1)]
    return "\n\n".join(pages)


def build_prompt(guideline_text: str) -> str:
    checks = "\n".join(
        f"  - {key}: params {json.dumps(params)}" for key, params in TECHNICAL_CHECKS.items()
    )
    return f"""Extract the video-relevant rules from the brand guideline below.

Return a JSON array. Each element has these fields:
  id            short unique snake_case slug
  title         short human title
  category      one of: technical, structure, logo, color, typography, text, captions, audio, accessibility, voice
  description   the requirement in one or two plain sentences, with the exact numbers from the guideline
  source_quote  short verbatim excerpt from the guideline
  severity      critical | major | minor  (critical = would make the asset unusable or unsafe, minor = polish)
  check_type    "technical" or "ai"
  check_key     for technical rules only, exactly one of the keys below
  params        for technical rules only, the values for that key's params
  criterion     for ai rules only: a single yes/no question a reviewer can answer by watching the video; \
phrase it so that "yes" means the video COMPLIES
  ai_scope      for ai rules only: "frames" (a few sampled frames are enough), "audio", or "video" (needs timing or motion)

Use check_type "technical" ONLY when the rule is one of these measurable checks:
{checks}
Everything else is check_type "ai". Split compound sentences into separate rules so each rule tests one thing. \
Do not invent rules that are not in the guideline.

GUIDELINE:
{guideline_text}
"""


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "rule"


def normalize_rules(raw: list[dict]) -> list[Rule]:
    """Validate model output, repair what is repairable, and drop what is not."""
    rules: list[Rule] = []
    seen: set[str] = set()
    for item in raw:
        try:
            rule = Rule.model_validate(item)
        except Exception:
            continue
        if rule.check_type == CheckType.technical and rule.check_key not in TECHNICAL_CHECKS:
            # The evaluator cannot run an unknown technical check, so let the video LLM judge it.
            rule.check_type = CheckType.ai
            rule.check_key = None
            rule.criterion = rule.criterion or rule.description
            rule.ai_scope = AiScope.video
        if rule.check_type == CheckType.ai and not rule.criterion:
            rule.criterion = rule.description
        rule.id = _slug(rule.id)
        base, n = rule.id, 2
        while rule.id in seen:
            rule.id = f"{base}_{n}"
            n += 1
        seen.add(rule.id)
        rules.append(rule)
    return rules


def extract_rules(pdf_path: str | Path, brand_name: str, llm: LLM) -> Brand:
    text = read_pdf_text(pdf_path)
    if not text.strip():
        raise ValueError(f"No text could be read from {pdf_path}; scanned PDFs are not supported yet.")
    raw = llm.generate_json(SYSTEM_PROMPT, build_prompt(text))
    if isinstance(raw, dict):  # some models wrap the array, e.g. {"rules": [...]}
        raw = next((v for v in raw.values() if isinstance(v, list)), [])
    rules = normalize_rules(raw)
    if not rules:
        raise ValueError("The model returned no usable rules.")
    return Brand(name=brand_name, source_file=Path(pdf_path).name, rules=rules)
