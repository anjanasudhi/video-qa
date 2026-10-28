"""Deterministic checks. Each takes (rule params, VideoContext) and returns (passed, detail, timestamps)."""

from dataclasses import dataclass, field
from pathlib import Path

from .probe import VideoInfo, mean_luminance, measure_audio, probe

CheckOutcome = tuple[bool, str, list[float]]


@dataclass
class VideoContext:
    path: Path
    info: VideoInfo
    _audio: dict | None = field(default=None, repr=False)
    _luma = None

    @classmethod
    def open(cls, path: str | Path) -> "VideoContext":
        return cls(Path(path), probe(path))

    @property
    def audio(self) -> dict:
        if self._audio is None:
            self._audio = measure_audio(self.path) if self.info.has_audio else {}
        return self._audio

    @property
    def luma(self):
        if self._luma is None:
            self._luma = mean_luminance(self.path, self.info.fps)
        return self._luma


def _ratio(text: str) -> float:
    a, b = text.split(":")
    return float(a) / float(b)


def aspect_ratio(p: dict, ctx: VideoContext) -> CheckOutcome:
    actual = ctx.info.width / ctx.info.height
    ok = any(abs(actual - _ratio(a)) / _ratio(a) < 0.01 for a in p["allowed"])
    return ok, f"{ctx.info.width}x{ctx.info.height} (ratio {actual:.2f}); allowed {', '.join(p['allowed'])}", []


def min_resolution(p: dict, ctx: VideoContext) -> CheckOutcome:
    short = min(ctx.info.width, ctx.info.height)
    return short >= p["min_short_side"], f"short side is {short}px; minimum {p['min_short_side']}px", []


def frame_rate(p: dict, ctx: VideoContext) -> CheckOutcome:
    ok = any(abs(ctx.info.fps - a) < 0.05 for a in p["allowed"])
    return ok, f"{ctx.info.fps:.2f} fps; allowed {p['allowed']}", []


def duration(p: dict, ctx: VideoContext) -> CheckOutcome:
    tol = p.get("tolerance", 0)
    ok = any(abs(ctx.info.duration - a) <= tol for a in p["allowed_seconds"])
    return ok, f"{ctx.info.duration:.1f}s; allowed {p['allowed_seconds']} (±{tol}s)", []


def loudness(p: dict, ctx: VideoContext) -> CheckOutcome:
    if "lufs" not in ctx.audio:
        return False, "no audio track", []
    lufs = ctx.audio["lufs"]
    ok = abs(lufs - p["target_lufs"]) <= p["tolerance"]
    return ok, f"{lufs:.1f} LUFS; target {p['target_lufs']} ±{p['tolerance']}", []


def true_peak(p: dict, ctx: VideoContext) -> CheckOutcome:
    if "true_peak" not in ctx.audio:
        return False, "no audio track", []
    peak = ctx.audio["true_peak"]
    return peak <= p["max_dbtp"], f"true peak {peak:.1f} dBTP; maximum {p['max_dbtp']} dBTP", []


def flashing(p: dict, ctx: VideoContext) -> CheckOutcome:
    """Counts opposing brightness swings of 10%+ per second, a simplified WCAG general-flash test."""
    luma, fps = ctx.luma, ctx.info.fps
    deltas = luma[1:] - luma[:-1]
    swings = [(i, 1 if d > 0 else -1) for i, d in enumerate(deltas) if abs(d) >= 0.1 * 255]
    flashes = [i for (i, s), (_, prev) in zip(swings[1:], swings[:-1]) if s != prev]
    window = int(round(fps))
    worst, worst_start = 0, 0
    for i in flashes:
        count = sum(1 for j in flashes if i <= j < i + window)
        if count > worst:
            worst, worst_start = count, i
    ok = worst <= p["max_per_second"]
    detail = f"up to {worst} flashes in one second; maximum {p['max_per_second']}"
    return ok, detail, [] if ok else [round(worst_start / fps, 1)]


CHECKS = {
    "aspect_ratio": aspect_ratio,
    "min_resolution": min_resolution,
    "frame_rate": frame_rate,
    "duration": duration,
    "loudness": loudness,
    "true_peak": true_peak,
    "flashing": flashing,
}
