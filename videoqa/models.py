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
    # Colours below are hex strings like "#5B2EFF", or objects {"name", "hex"} / {"name", "rgb": [r, g, b]} /
    # {"name", "cmyk": [c, m, y, k]} with CMYK in percent. Copy them exactly as the guideline writes them.
    "palette_adherence": {"palette": "list of approved colours", "tolerance": "ΔE, default 12", "max_off_palette_share": "0-1, default 0.08"},
    "banned_hue": {"hue_ranges": "list of [min, max] hue degrees to forbid, e.g. [[80, 170]] for green", "max_share": "0-1, default 0.01"},
    "color_coverage": {"color": "one colour", "max_share": "largest allowed share of the frame, 0-1"},
    "text_height": {"min_fraction": "minimum text height as a fraction of frame height, e.g. 0.04"},
    "text_lines": {"max_lines": "maximum lines in a block of on-screen text"},
    "text_words_per_line": {"max_words": "maximum words per line of on-screen text"},
    "text_safe_area": {"margin": "fraction of each edge that must stay clear of text, e.g. 0.1"},
    "text_contrast": {"min_ratio": "minimum WCAG contrast ratio, e.g. 4.5"},
    "text_color": {"on_dark": "required text colour on dark backgrounds", "on_light": "required text colour on light backgrounds"},
    "text_present": {"text": "exact wording that must appear", "last_seconds": "window at the end of the video, e.g. 3"},
    "banned_words": {"words": "words or phrases that must not appear in on-screen text"},
    "logo_first_seen": {"seconds": "the logo must be visible by this time"},
    "logo_bug_corner": {"corner": "top-left | top-right | bottom-left | bottom-right", "safe_margin": "fraction of edge, e.g. 0.05", "end_card_seconds": "length of the end card, e.g. 2"},
    "logo_bug_width": {"min_fraction": "smallest width as a fraction of frame width", "max_fraction": "largest width", "end_card_seconds": "e.g. 2"},
    "logo_bug_absent_on_end_card": {"end_card_seconds": "e.g. 2"},
    "end_card": {"seconds": "end card length", "background_colors": "list of allowed solid background colours"},
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
    logo_files: list[str] = Field(default_factory=list, description="Logo images (transparent PNG) the logo checks match against")
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
