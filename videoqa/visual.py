"""Pixel-level analysis shared by the deterministic visual checks: sampled frames, colour maths, OCR and logo detection."""

import subprocess
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import cv2
import numpy as np

from .probe import VideoInfo

LONG_SIDE = 960  # analysis resolution (longer side, px)
MAX_SAMPLE_FPS = 2.0
MAX_FRAMES = 150


class MissingAsset(Exception):
    """A check needs a file the brand does not have yet (for example its logo)."""


@dataclass
class TextBox:
    text: str
    confidence: float
    x0: float  # all coordinates are fractions of the frame, 0-1
    y0: float
    x1: float
    y1: float

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def width(self) -> float:
        return self.x1 - self.x0


@dataclass
class LogoHit:
    score: float
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def center(self) -> tuple[float, float]:
        return (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2


# ---------- colour helpers ----------

def hex_to_rgb(value: str) -> np.ndarray:
    value = value.lstrip("#")
    return np.array([int(value[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.uint8)


CMYK_EXTRA_TOLERANCE = 8.0  # ΔE added when a colour is only known in CMYK, which converts to screen colour approximately


def cmyk_to_rgb(c: float, m: float, y: float, k: float, profile: str | None = None) -> np.ndarray:
    """CMYK (percent, 0-100) to sRGB. Uses an ICC profile when one is given, otherwise the standard formula."""
    if profile:
        from PIL import Image, ImageCms

        transform = ImageCms.buildTransform(ImageCms.getOpenProfile(profile), ImageCms.createProfile("sRGB"), "CMYK", "RGB")
        px = Image.new("CMYK", (1, 1), tuple(int(round(v * 2.55)) for v in (c, m, y, k)))
        return np.array(ImageCms.applyTransform(px, transform).getpixel((0, 0)), dtype=np.uint8)
    c, m, y, k = (v / 100.0 for v in (c, m, y, k))
    return np.round(255 * np.array([(1 - c) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k)])).astype(np.uint8)


def parse_color(entry, cmyk_profile: str | None = None) -> tuple[np.ndarray, bool, str]:
    """Reads a colour from a rule: '#5B2EFF', {'hex': ...}, {'rgb': [r, g, b]} or {'cmyk': [c, m, y, k]} (percent).

    Returns (sRGB, approximate, label). When an entry has both RGB/hex and CMYK, RGB/hex wins because it is exact.
    approximate is True when the colour had to be converted from CMYK.
    """
    if isinstance(entry, str):
        entry = {"hex": entry}
    label = entry.get("name") or ""
    if entry.get("hex"):
        rgb = hex_to_rgb(entry["hex"])
        return rgb, False, label or to_hex(rgb)
    if entry.get("rgb"):
        rgb = np.array(entry["rgb"], dtype=np.uint8)
        return rgb, False, label or to_hex(rgb)
    if entry.get("cmyk"):
        rgb = cmyk_to_rgb(*entry["cmyk"], profile=cmyk_profile)
        return rgb, True, label or "CMYK " + "/".join(f"{v:g}" for v in entry["cmyk"])
    raise ValueError(f"colour needs hex, rgb or cmyk: {entry!r}")


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8).astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def delta_e(lab_a: np.ndarray, lab_b: np.ndarray) -> np.ndarray:
    """CIE76 colour difference: Euclidean distance in L*a*b*. About 2 is the just-noticeable difference."""
    return np.linalg.norm(lab_a - lab_b, axis=-1)


def relative_luminance(rgb) -> float:
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


def contrast_ratio(rgb_a, rgb_b) -> float:
    la, lb = relative_luminance(rgb_a), relative_luminance(rgb_b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def to_hex(rgb) -> str:
    return "#%02X%02X%02X" % tuple(int(v) for v in rgb)


# ---------- analysis ----------

class VisualAnalysis:
    """Lazily computed, cached measurements for one video."""

    def __init__(self, path: Path, info: VideoInfo, logo_paths: list[Path] | None = None):
        self.path = Path(path)
        self.info = info
        self.logo_paths = [Path(p) for p in logo_paths or []]
        self.sample_fps = min(MAX_SAMPLE_FPS, MAX_FRAMES / max(info.duration, 0.1))

    @cached_property
    def frames(self) -> np.ndarray:
        info = self.info
        if info.width >= info.height:
            w, h = LONG_SIDE, int(round(LONG_SIDE * info.height / info.width / 2)) * 2
        else:
            h, w = LONG_SIDE, int(round(LONG_SIDE * info.width / info.height / 2)) * 2
        out = subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", str(self.path), "-vf", f"fps={self.sample_fps},scale={w}:{h}",
             "-frames:v", str(MAX_FRAMES), "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
            capture_output=True, check=True,
        ).stdout
        data = np.frombuffer(out, dtype=np.uint8)
        n = len(data) // (w * h * 3)
        return data[: n * w * h * 3].reshape(n, h, w, 3)

    @cached_property
    def times(self) -> list[float]:
        return [i / self.sample_fps for i in range(len(self.frames))]

    @property
    def duration(self) -> float:
        return self.info.duration

    def end_card_indices(self, seconds: float) -> list[int]:
        """Frames inside the closing window, skipping the first half second so a transition is not judged."""
        start = self.duration - seconds + 0.5
        return [i for i, t in enumerate(self.times) if t >= start]

    def body_indices(self, seconds: float) -> list[int]:
        end = self.duration - seconds
        return [i for i, t in enumerate(self.times) if t < end]

    # -- OCR --

    @cached_property
    def _ocr_engine(self):
        from rapidocr_onnxruntime import RapidOCR

        return RapidOCR()

    @cached_property
    def ocr_indices(self) -> list[int]:
        step = max(1, int(round(self.sample_fps)))  # about one OCR pass per second
        return list(range(0, len(self.frames), step))

    @cached_property
    def ocr(self) -> dict[int, list[TextBox]]:
        results: dict[int, list[TextBox]] = {}
        for i in self.ocr_indices:
            frame = self.frames[i]
            h, w = frame.shape[:2]
            raw, _ = self._ocr_engine(np.ascontiguousarray(frame[..., ::-1]))
            boxes = []
            for points, text, conf in raw or []:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                boxes.append(TextBox(text.strip(), float(conf), min(xs) / w, min(ys) / h, max(xs) / w, max(ys) / h))
            results[i] = boxes
        return results

    # -- logo --

    @cached_property
    def _logos(self) -> list[np.ndarray]:
        if not self.logo_paths:
            raise MissingAsset("this check needs the brand's logo file; upload it in the Admin tab")
        logos = []
        for path in self.logo_paths:
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise MissingAsset(f"could not read logo file {path.name}")
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            logos.append(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))
        return logos

    @cached_property
    def logo_hits(self) -> dict[int, list[LogoHit]]:
        return {i: detect_logos(self.frames[i], self._logos) for i in self.ocr_indices}


# ---------- logo detection ----------

COARSE_WIDTH = 320
FINE_WIDTH = 640
COARSE_SCALES = np.geomspace(0.06, 0.55, 16)  # logo width as a fraction of frame width
COARSE_MIN_SCORE = 0.45
LOGO_MIN_SCORE = 0.82
MIN_TEMPLATE_STD = 12  # a logo that is nearly invisible on this background cannot be matched reliably


def _dominant_colors(rgb: np.ndarray, k: int = 2) -> list[np.ndarray]:
    quant = rgb[::4, ::4].reshape(-1, 3) // 16
    keys, counts = np.unique(quant, axis=0, return_counts=True)
    return [(keys[i] * 16 + 8).astype(np.float32) for i in np.argsort(-counts)[:k]]


def _composite(logo: np.ndarray, bg: np.ndarray, tw: int, th: int) -> np.ndarray:
    small = cv2.resize(logo, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32)
    alpha = small[..., 3:4] / 255.0
    return (small[..., :3] * alpha + bg * (1 - alpha)).astype(np.uint8)


def _template_size(logo: np.ndarray, frame_w: int, width_frac: float) -> tuple[int, int]:
    tw = int(frame_w * width_frac)
    return tw, int(round(logo.shape[0] * tw / logo.shape[1]))


def _iou(a: LogoHit, b: LogoHit) -> float:
    ix = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    iy = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    inter = ix * iy
    union = (a.x1 - a.x0) * (a.y1 - a.y0) + (b.x1 - b.x0) * (b.y1 - b.y0) - inter
    return inter / union if union else 0.0


def _distinct(proposals: list, cw: int, ch: int) -> list:
    """Drops proposals that overlap a better-scoring one, so look-alike peaks cannot crowd out the rest."""
    kept: list[tuple[LogoHit, tuple]] = []
    for prop in proposals:
        _, wf, xf, yf, logo, _ = prop
        tw, th = _template_size(logo, cw, wf)
        box = LogoHit(0, xf, yf, xf + tw / cw, yf + th / ch)
        if all(_iou(box, other) < 0.7 for other, _ in kept):
            kept.append((box, prop))
    return [prop for _, prop in kept]


def detect_logos(frame: np.ndarray, logos: list[np.ndarray]) -> list[LogoHit]:
    """Finds every placement of the logo, small or large.

    The transparent logo is composited on each dominant background colour before matching, which reproduces how
    a logo is normally overlaid on flat graphics (it will not find logos over busy footage). A fast greyscale
    pass proposes candidates; each is then verified in colour at a finer scale so look-alike shapes are rejected.
    """
    h0, w0 = frame.shape[:2]
    coarse = cv2.resize(frame, (COARSE_WIDTH, int(h0 * COARSE_WIDTH / w0)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(coarse, cv2.COLOR_RGB2GRAY)
    ch, cw = gray.shape

    proposals = []  # (score, width_frac, x_frac, y_frac, logo, bg)
    for bg in _dominant_colors(coarse):
        for logo in logos:
            for wf in COARSE_SCALES:
                tw, th = _template_size(logo, cw, wf)
                if th < 6 or th >= ch or tw >= cw:
                    continue
                template = cv2.cvtColor(_composite(logo, bg, tw, th), cv2.COLOR_RGB2GRAY)
                if template.std() < MIN_TEMPLATE_STD:
                    continue
                res = np.nan_to_num(cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED), nan=0.0, posinf=0.0, neginf=0.0)
                for _ in range(2):
                    _, score, _, (x, y) = cv2.minMaxLoc(res)
                    if score < COARSE_MIN_SCORE:
                        break
                    proposals.append((score, wf, x / cw, y / ch, logo, bg))
                    res[max(0, y - th) : y + th, max(0, x - tw) : x + tw] = 0
    proposals.sort(key=lambda p: -p[0])
    proposals = _distinct(proposals, cw, ch)[:60]

    fine = cv2.resize(frame, (FINE_WIDTH, int(h0 * FINE_WIDTH / w0)), interpolation=cv2.INTER_AREA)
    fh, fw = fine.shape[:2]
    hits: list[LogoHit] = []
    for _, wf, xf, yf, logo, bg in proposals:
        best = None
        for wf2 in np.linspace(wf * 0.88, wf * 1.12, 17):
            tw, th = _template_size(logo, fw, wf2)
            if th < 8 or th >= fh or tw >= fw:
                continue
            pad = max(10, int(tw * 0.25))
            x0, y0 = int(xf * fw), int(yf * fh)
            rx0, ry0 = max(0, x0 - pad), max(0, y0 - pad)
            rx1, ry1 = min(fw, x0 + tw + pad), min(fh, y0 + th + pad)
            roi = fine[ry0:ry1, rx0:rx1]
            if roi.shape[0] < th or roi.shape[1] < tw:
                continue
            template = _composite(logo, bg, tw, th)
            if template.std() < MIN_TEMPLATE_STD:
                continue
            res = np.nan_to_num(cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED), nan=0.0, posinf=0.0, neginf=0.0)
            _, score, _, (lx, ly) = cv2.minMaxLoc(res)
            if best is None or score > best[0]:
                best = (float(score), (rx0 + lx) / fw, (ry0 + ly) / fh, tw / fw, th / fh)
        if best is None or best[0] < LOGO_MIN_SCORE:
            continue
        hit = LogoHit(best[0], best[1], best[2], best[1] + best[3], best[2] + best[4])
        if all(_iou(hit, other) < 0.3 for other in hits):
            hits.append(hit)
    return hits
