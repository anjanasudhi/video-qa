import json
from functools import cache
from pathlib import Path

import pytest

from videoqa.evaluate import evaluate, score
from videoqa.models import Brand, CheckType, Rule, Status
from videoqa.vision import MockReviewer, build_review_prompt

ROOT = Path(__file__).resolve().parent.parent
VIDEOS = ROOT / "samples" / "videos"
MANIFEST = json.loads((VIDEOS / "manifest.json").read_text())
BRAND = Brand(
    name="Lumora",
    rules=[Rule.model_validate(r) for r in json.loads((ROOT / "samples/rules/lumora_expected_rules.json").read_text())],
    logo_files=[str(p) for p in sorted((ROOT / "samples/assets").glob("lumora_logo_*.png"))],
    font_files=[str(p) for p in sorted((ROOT / "samples/assets/fonts").glob("*.otf"))],
)
TECHNICAL_IDS = {r.id for r in BRAND.rules if r.check_type == CheckType.technical}


@cache
def deterministic_report(video: str):
    return evaluate(VIDEOS / video, BRAND)


@pytest.mark.parametrize("video", sorted(MANIFEST))
def test_deterministic_checks_match_ground_truth(video):
    """Specs, colours, on-screen text and logo placement are all measured, so they must match the planted violations exactly."""
    report = deterministic_report(video)
    failed = {r.rule_id for r in report.failures}
    assert failed == set(MANIFEST[video]["expected_violations"]) & TECHNICAL_IDS
    assert not [r for r in report.results if r.status in (Status.error, Status.skipped) and r.check_type == CheckType.technical]


def test_logo_rules_are_skipped_without_a_logo_file():
    no_logo = BRAND.model_copy(update={"logo_files": []})
    report = evaluate(VIDEOS / "lumora_compliant_9x16_6s.mp4", no_logo)
    skipped = {r.rule_id for r in report.results if r.status == Status.skipped and r.check_type == CheckType.technical}
    assert skipped == {"brand_reveal", "logo_bug_position", "logo_bug_size", "logo_bug_end_card", "end_card"}


def test_failures_quote_measured_values():
    by_id = {r.rule_id: r for r in deterministic_report("lumora_violations_brand_16x9_15s.mp4").results}
    assert "bottom-left" in by_id["logo_bug_position"].detail
    assert "14.8%" in by_id["logo_bug_size"].detail  # the bug is drawn at 15% of the frame width
    assert "hue" in by_id["banned_color"].detail
    assert by_id["contrast"].detail.startswith("contrast 2.")


def test_ai_rules_are_skipped_without_reviewer():
    report = deterministic_report("lumora_compliant_9x16_6s.mp4")
    skipped = {r.rule_id for r in report.results if r.status == Status.skipped}
    assert skipped == {r.id for r in BRAND.rules} - TECHNICAL_IDS


def test_ai_verdicts_feed_the_report():
    verdicts = {r.id: {"complies": True, "confidence": 0.9, "explanation": "ok"} for r in BRAND.rules if r.check_type == CheckType.ai}
    verdicts["captions"] = {"complies": False, "confidence": 0.8, "explanation": "No captions seen", "timestamps": [1.0, 5.5]}
    del verdicts["music_tone"]  # reviewer forgot one
    reviewer = MockReviewer(verdicts)
    report = evaluate(VIDEOS / "lumora_compliant_16x9_15s.mp4", BRAND, reviewer)

    assert len(reviewer.calls) == 1  # all AI rules in a single request
    by_id = {r.rule_id: r for r in report.results}
    assert by_id["captions"].status == Status.failed and by_id["captions"].timestamps == [1.0, 5.5]
    assert "80%" in by_id["captions"].detail
    assert by_id["music_tone"].status == Status.error
    assert by_id["logo_bug_position"].status == Status.passed
    assert 0 < report.score < 100


def test_reviewer_crash_becomes_errors_not_exception():
    class Boom:
        def review(self, *a):
            raise RuntimeError("quota exceeded")

    report = evaluate(VIDEOS / "lumora_compliant_16x9_15s.mp4", BRAND, Boom())
    assert all(r.status == Status.error and "quota exceeded" in r.detail for r in report.results if r.check_type == CheckType.ai)


def test_score_weights_severity():
    from videoqa.models import CheckResult, Severity

    def res(sev, status):
        return CheckResult(rule_id="x", title="x", severity=sev, check_type=CheckType.ai, status=status)

    assert score([res(Severity.critical, Status.passed), res(Severity.minor, Status.failed)]) == 75.0
    assert score([res(Severity.major, Status.skipped)]) == 0.0


def test_prompt_numbers_every_rule():
    from videoqa.probe import probe

    info = probe(VIDEOS / "lumora_compliant_16x9_15s.mp4")
    ai_rules = [r for r in BRAND.rules if r.check_type == CheckType.ai]
    prompt = build_review_prompt(info, ai_rules)
    assert "1920x1080" in prompt and f"{len(ai_rules)}. [" in prompt and "[captions]" in prompt


def test_missing_font_file_falls_back_to_the_ai_criterion():
    no_font = BRAND.model_copy(update={"font_files": []})
    reviewer = MockReviewer({"font": {"complies": False, "confidence": 0.9, "explanation": "Serif headline", "timestamps": [1.0]}})
    report = evaluate(VIDEOS / "lumora_compliant_9x16_6s.mp4", no_font, reviewer)
    font = next(r for r in report.results if r.rule_id == "font")
    assert font.status == Status.failed and font.detail.startswith("AI judgement, since the brand file")
    assert "font" in reviewer.calls[0]
    # without a reviewer the rule is skipped, never guessed
    report = evaluate(VIDEOS / "lumora_compliant_9x16_6s.mp4", no_font)
    assert next(r for r in report.results if r.rule_id == "font").status == Status.skipped
