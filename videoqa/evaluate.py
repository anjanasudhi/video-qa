"""Run a brand's rules against a video and assemble a compliance report."""

from pathlib import Path

from .models import Brand, CheckResult, CheckType, Report, Rule, Severity, Status
from .store import logo_paths
from .technical import CHECKS, VideoContext
from .visual import MissingAsset
from .vision import VideoReviewer

WEIGHTS = {Severity.critical: 3, Severity.major: 2, Severity.minor: 1}


def _result(rule: Rule, status: Status, detail: str = "", timestamps: list[float] | None = None) -> CheckResult:
    return CheckResult(
        rule_id=rule.id, title=rule.title, severity=rule.severity, check_type=rule.check_type,
        status=status, detail=detail, timestamps=timestamps or [],
    )


def run_technical(rule: Rule, ctx: VideoContext) -> CheckResult:
    check = CHECKS.get(rule.check_key or "")
    if check is None:
        return _result(rule, Status.error, f"unknown technical check '{rule.check_key}'")
    try:
        ok, detail, timestamps = check(rule.params, ctx)
    except MissingAsset as exc:
        return _result(rule, Status.skipped, str(exc))
    except Exception as exc:  # a malformed param set should not sink the whole report
        return _result(rule, Status.error, f"{type(exc).__name__}: {exc}")
    return _result(rule, Status.passed if ok else Status.failed, detail, timestamps)


def run_ai(rules: list[Rule], ctx: VideoContext, reviewer: VideoReviewer) -> list[CheckResult]:
    try:
        verdicts = {v.get("rule_id"): v for v in reviewer.review(ctx.path, ctx.info, rules)}
    except Exception as exc:
        return [_result(r, Status.error, f"AI review failed: {exc}") for r in rules]
    results = []
    for rule in rules:
        v = verdicts.get(rule.id)
        if v is None or not isinstance(v.get("complies"), bool):
            results.append(_result(rule, Status.error, "the reviewer gave no verdict for this rule"))
            continue
        confidence = v.get("confidence")
        note = f" (confidence {confidence:.0%})" if isinstance(confidence, (int, float)) else ""
        timestamps = [float(t) for t in v.get("timestamps") or [] if isinstance(t, (int, float))]
        results.append(_result(rule, Status.passed if v["complies"] else Status.failed,
                               f"{v.get('explanation', '')}{note}".strip(), timestamps))
    return results


def score(results: list[CheckResult]) -> float:
    judged = [r for r in results if r.status in (Status.passed, Status.failed)]
    total = sum(WEIGHTS[r.severity] for r in judged)
    if not total:
        return 0.0
    return round(100 * sum(WEIGHTS[r.severity] for r in judged if r.status == Status.passed) / total, 1)


def evaluate(video: str | Path, brand: Brand, reviewer: VideoReviewer | None = None) -> Report:
    """Technical rules always run. AI rules run only when a reviewer is given; otherwise they are skipped."""
    ctx = VideoContext.open(video, brand.name, logo_paths(brand))
    results: dict[str, CheckResult] = {}
    ai_rules: list[Rule] = []
    for rule in (r for r in brand.rules if r.enabled):
        if rule.check_type == CheckType.technical:
            results[rule.id] = run_technical(rule, ctx)
        elif reviewer is None:
            results[rule.id] = _result(rule, Status.skipped, "no AI reviewer configured")
        else:
            ai_rules.append(rule)
    if ai_rules:
        for res in run_ai(ai_rules, ctx, reviewer):
            results[res.rule_id] = res
    ordered = [results[r.id] for r in brand.rules if r.id in results]
    return Report(brand=brand.name, video=Path(video).name, score=score(ordered), results=ordered)
