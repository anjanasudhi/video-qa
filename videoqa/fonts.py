"""Deterministic typeface check: compares the shape of every letter in the video with the brand font file.

Each on-screen letter is cut out of a full-resolution frame and compared, as a normalised bitmap, with a rendering
of every letter in the brand font and in a set of other installed fonts. A line passes when most of its letters
are closer to the brand font than to any other font. This catches serif, script, monospace and clearly different
sans-serif faces. It needs every weight the guideline allows (for example Regular and Bold) to be uploaded, and
it cannot tell apart two nearly identical designs.
"""

import os
import string
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CHARS = string.ascii_letters + string.digits
GLYPH = 32  # glyphs are compared as GLYPH x GLYPH bitmaps
MAX_CONTROL_FONTS = 40
TIE_MARGIN = 0.02  # a letter counts for the brand font unless another font beats it by more than this

FONT_DIRS = [
    "/usr/share/fonts", "/usr/local/share/fonts", "~/.fonts", "~/.local/share/fonts",
    "/Library/Fonts", "/System/Library/Fonts", "C:/Windows/Fonts",
]


def _normalise(binary: np.ndarray) -> np.ndarray:
    """Tight 0/255 bitmap to a centred GLYPH x GLYPH float vector, keeping the aspect ratio."""
    h, w = binary.shape
    scale = (GLYPH - 2) / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    small = cv2.resize(binary, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    out = np.zeros((GLYPH, GLYPH), np.float32)
    y0, x0 = (GLYPH - nh) // 2, (GLYPH - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = small
    return out.ravel()


@lru_cache(maxsize=256)
def font_templates(path: str) -> np.ndarray:
    """One normalised bitmap per letter and digit of the font, shape (n, GLYPH*GLYPH)."""
    font = ImageFont.truetype(path, 120)
    rows = []
    for ch in CHARS:
        img = Image.new("L", (240, 240), 0)
        ImageDraw.Draw(img).text((30, 20), ch, font=font, fill=255)
        arr = (np.array(img) > 127).astype(np.uint8) * 255
        ys, xs = np.where(arr > 0)
        if len(xs):
            rows.append(_normalise(arr[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]))
    if not rows:
        raise ValueError(f"{Path(path).name} has no Latin glyphs to compare against")
    return np.stack(rows)


def font_name(path: str) -> str:
    family, style = ImageFont.truetype(path, 20).getname()
    return f"{family} {style}".strip()


def _has_latin(font: ImageFont.FreeTypeFont) -> bool:
    return font.getmask("a").getbbox() is not None and bytes(font.getmask("a")) != bytes(font.getmask("\uffff"))


@lru_cache(maxsize=8)
def control_fonts(exclude_families: tuple[str, ...]) -> tuple[str, ...]:
    """Installed fonts to compare against, never including the brand's own family."""
    found: dict[tuple[str, str], str] = {}
    for root in FONT_DIRS:
        root_path = Path(os.path.expanduser(root))
        if not root_path.exists():
            continue
        for path in sorted(root_path.rglob("*")):
            if path.suffix.lower() not in (".ttf", ".otf"):
                continue
            try:
                font = ImageFont.truetype(str(path), 40)
                family, style = font.getname()
                if family in exclude_families or not _has_latin(font):
                    continue
            except Exception:
                continue
            found.setdefault((family, style), str(path))
    fonts = [found[k] for k in sorted(found)]
    if len(fonts) > MAX_CONTROL_FONTS:
        step = len(fonts) / MAX_CONTROL_FONTS
        fonts = [fonts[int(i * step)] for i in range(MAX_CONTROL_FONTS)]
    return tuple(fonts)


def _best_overlap(glyphs: np.ndarray, templates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inter = glyphs @ templates.T
    union = glyphs.sum(1)[:, None] + templates.sum(1)[None, :] - inter
    iou = inter / np.maximum(union, 1e-6)
    return iou.max(axis=1), iou.argmax(axis=1)


def letters_in(crop: np.ndarray) -> list[np.ndarray]:
    """Cuts the individual letters out of a text crop (RGB) as normalised bitmaps."""
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if mask.mean() > 127:  # text is the minority colour
        mask = 255 - mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    height, width = crop.shape[:2]
    out = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if h < 0.3 * height or h > 1.05 * height or w > 1.6 * height or area < 30:
            continue  # dots, specks, merged blobs
        if x == 0 or y == 0 or x + w >= width or y + h >= height:
            continue  # cut off by the crop
        out.append(_normalise((labels[y : y + h, x : x + w] == i).astype(np.uint8) * 255))
    return out


@dataclass
class LineResult:
    text: str
    time: float
    letters: int
    brand_share: float
    looks_like: str  # the other font most of the letters resemble


def compare_line(crop: np.ndarray, brand: np.ndarray, control: np.ndarray, control_names: list[str], control_owner: np.ndarray):
    letters = letters_in(crop)
    if len(letters) < 6:
        return None
    glyphs = np.stack(letters)
    brand_score, _ = _best_overlap(glyphs, brand)
    other_score, other_idx = _best_overlap(glyphs, control)
    share = float((brand_score >= other_score - TIE_MARGIN).mean())
    losing = control_owner[other_idx[brand_score < other_score - TIE_MARGIN]]
    looks_like = control_names[int(np.bincount(losing).argmax())] if len(losing) else ""
    return len(letters), share, looks_like


def load_references(brand_fonts: list[Path]):
    brand = np.concatenate([font_templates(str(p)) for p in brand_fonts])
    families = tuple({ImageFont.truetype(str(p), 20).getname()[0] for p in brand_fonts})
    paths = control_fonts(families)
    if len(paths) < 3:
        raise ValueError("fewer than three other fonts are installed to compare against")
    control = np.concatenate([font_templates(p) for p in paths])
    owner = np.concatenate([np.full(len(font_templates(p)), i) for i, p in enumerate(paths)])
    return brand, control, [font_name(p) for p in paths], owner, families


def hires_frame(video: Path, seconds: float, width: int, height: int, long_side: int = 1920) -> np.ndarray:
    """One frame at up to long_side resolution, for reading small text."""
    scale = min(1.0, long_side / max(width, height))
    w, h = int(round(width * scale / 2)) * 2, int(round(height * scale / 2)) * 2
    out = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-ss", f"{seconds:.3f}", "-i", str(video), "-frames:v", "1",
         "-vf", f"scale={w}:{h}", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(out, dtype=np.uint8).reshape(h, w, 3)
