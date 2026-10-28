from pathlib import Path

import pytest

from videoqa.extract import build_prompt, extract_rules, normalize_rules, read_pdf_text
from videoqa.llm import MockLLM
from videoqa.models import CheckType
from videoqa.store import list_brands, load_brand, save_brand

ROOT = Path(__file__).resolve().parent.parent
PDF = ROOT / "samples" / "guidelines" / "lumora_brand_guidelines.pdf"
FIXTURE = ROOT / "samples" / "rules" / "lumora_expected_rules.json"


def test_pdf_text_includes_video_section():
    text = read_pdf_text(PDF)
    assert "6.4 Logo bug" in text
    assert "−16 LUFS" in text


def test_prompt_lists_technical_checks_and_guideline():
    prompt = build_prompt("GUIDELINE BODY")
    assert "min_resolution" in prompt and "GUIDELINE BODY" in prompt


def test_extract_with_mock_llm_round_trips(tmp_path):
    llm = MockLLM(fixture=FIXTURE)
    brand = extract_rules(PDF, "Lumora", llm)
    assert len(brand.rules) == 27
    assert len(llm.calls) == 1 and "6.4 Logo bug" in llm.calls[0][1]

    save_brand(brand, tmp_path)
    assert list_brands(tmp_path) == ["Lumora"]
    assert load_brand("lumora", tmp_path) == brand


def test_every_manifest_violation_has_a_rule():
    import json

    manifest = json.loads((ROOT / "samples" / "videos" / "manifest.json").read_text())
    brand = extract_rules(PDF, "Lumora", MockLLM(fixture=FIXTURE))
    ids = {r.id for r in brand.rules}
    for name, entry in manifest.items():
        assert set(entry["expected_violations"]) <= ids, name


def test_normalize_repairs_and_dedupes():
    raw = [
        {"id": "Bad Check", "title": "t", "category": "audio", "description": "Keep music quiet",
         "check_type": "technical", "check_key": "made_up", "params": {}},
        {"id": "bad check", "title": "t", "category": "logo", "description": "Logo on the left",
         "check_type": "ai"},
        {"title": "no id or type"},
    ]
    rules = normalize_rules(raw)
    assert [r.id for r in rules] == ["bad_check", "bad_check_2"]
    assert rules[0].check_type == CheckType.ai and rules[0].criterion == "Keep music quiet"
    assert rules[1].criterion == "Logo on the left"


def test_wrapped_response_is_unwrapped():
    llm = MockLLM(response={"rules": [{"id": "x", "title": "t", "category": "logo", "description": "d", "check_type": "ai"}]})
    assert [r.id for r in extract_rules(PDF, "B", llm).rules] == ["x"]


def test_empty_response_raises():
    with pytest.raises(ValueError):
        extract_rules(PDF, "B", MockLLM(response=[]))


def test_case_only_banned_words_become_an_ai_rule():
    raw = [{"id": "casing", "title": "t", "category": "voice", "description": "Write Lumora, never LUMORA",
            "check_type": "technical", "check_key": "banned_words", "params": {"words": ["LUMORA", "lumora"]}}]
    (rule,) = normalize_rules(raw)
    assert rule.check_type == CheckType.ai and rule.criterion == "Write Lumora, never LUMORA"
