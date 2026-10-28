from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Severity(str, Enum):
    critical = "critical"
    major = "major"
    minor = "minor"


class CheckType(str, Enum):
    technical = "technical"  # deterministic, run with ffprobe / audio analysis
    ai = "ai"  # judged by a video LLM against `criterion`


class AiScope(str, Enum):
    frames = "frames"  # sampled video frames are enough
    audio = "audio"  # needs the soundtrack
    video = "video"  # needs motion / timing, send the whole clip


# Deterministic checks the evaluator knows how to run, with the params each one reads.
TECHNICAL_CHECKS: dict[str, dict[str, str]] = {
    "aspect_ratio": {"allowed": "list of 'W:H' strings, e.g. ['16:9', '9:16', '1:1']"},
    "min_resolution": {"min_short_side": "pixels on the shorter side of the frame"},
    "frame_rate": {"allowed": "list of fps values, e.g. [25, 30]"},
    "duration": {"allowed_seconds": "list of approved lengths", "tolerance": "seconds of slack"},
    "loudness": {"target_lufs": "integrated loudness target", "tolerance": "LU of slack"},
    "true_peak": {"max_dbtp": "maximum true peak in dBTP"},
    "flashing": {"max_per_second": "maximum luminance flashes per second"},
}


class Rule(BaseModel):
    id: str = Field(description="Short unique snake_case slug, e.g. logo_bug_position")
    title: str
    category: str = Field(description="One of: technical, structure, logo, color, typography, text, captions, audio, accessibility, voice")
    description: str = Field(description="The requirement in one or two plain sentences")
    source_quote: str = Field(default="", description="Short verbatim excerpt from the guideline this rule comes from")
    severity: Severity = Severity.major
    check_type: CheckType
    check_key: str | None = Field(default=None, description="Technical checks only: one of the known check keys")
    params: dict[str, Any] = Field(default_factory=dict)
    criterion: str = Field(default="", description="AI checks only: a yes/no question a reviewer can answer from the video")
    ai_scope: AiScope = AiScope.frames
    enabled: bool = True


class Brand(BaseModel):
    name: str
    source_file: str = ""
    rules: list[Rule] = Field(default_factory=list)


class Status(str, Enum):
    passed = "pass"
    failed = "fail"
    error = "error"  # the check could not be run
    skipped = "skipped"


class CheckResult(BaseModel):
    rule_id: str
    title: str
    severity: Severity
    check_type: CheckType
    status: Status
    detail: str = ""
    timestamps: list[float] = Field(default_factory=list, description="Seconds into the video where the issue shows")


class Report(BaseModel):
    brand: str
    video: str
    score: float = Field(description="0-100, severity-weighted share of evaluated rules that passed")
    results: list[CheckResult]

    @property
    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if r.status == Status.failed]
