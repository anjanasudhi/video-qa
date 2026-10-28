import json
from pathlib import Path

import pytest

from videoqa.evaluate import evaluate, score
from videoqa.models import Brand, CheckType, Rule, Status
from videoqa.vision import MockReviewer, build_review_prompt

ROOT = Path(__file__).resolve().parent.parent
VIDEOS = ROOT / "samples" / "videos"
MANIFEST = json.loads((VIDEOS / "manifest.json").read_text())
BRAND = Brand(name="Lumora", rules=[Rule.model_validate(r) for r in json.loads((ROOT / "samples/rules/lumora_expected_rules.json").read_text())])
TECHNICAL_IDS = {r.id for r in BRAND.rules if r.check_type == CheckType.technical}


@pytest.mark.parametrize("video", sorted(MANIFEST))
def test_technical_checks_match_ground_truth(video):
    report = evaluate(VIDEOS / video, BRAND)
    failed = {r.rule_id for r in report.failures}
    assert failed == set(MANIFEST[video]["expected_violations"]) & TECHNICAL_IDS
    assert not [r for r in report.results if r.status == Status.error]


def test_ai_rules_are_skipped_without_reviewer():
    report = evaluate(VIDEOS / "lumora_compliant_9x16_6s.mp4", BRAND)
    skipped = {r.rule_id for r in report.results if r.status == Status.skipped}
    assert skipped == {r.id for r in BRAND.rules} - TECHNICAL_IDS


def test_ai_verdicts_feed_the_report():
    verdicts = {r.id: {"complies": True, "confidence": 0.9, "explanation": "ok"} for r in BRAND.rules if r.check_type == CheckType.ai}
    verdicts["captions"] = {"complies": False, "confidence": 0.8, "explanation": "No captions seen", "timestamps": [1.0, 5.5]}
    del verdicts["tagline"]  # reviewer forgot one
    reviewer = MockReviewer(verdicts)
    report = evaluate(VIDEOS / "lumora_compliant_16x9_15s.mp4", BRAND, reviewer)

    assert len(reviewer.calls) == 1  # all AI rules in a single request
    by_id = {r.rule_id: r for r in report.results}
    assert by_id["captions"].status == Status.failed and by_id["captions"].timestamps == [1.0, 5.5]
    assert "80%" in by_id["captions"].detail
    assert by_id["tagline"].status == Status.error
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
