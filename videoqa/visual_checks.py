"""Deterministic checks on what is on screen: colours, on-screen text (via OCR) and logo placement.

Each check takes (rule params, VideoContext) and returns (passed, detail, timestamps), like the checks in
technical.py. They measure pixels, so the same video always gives the same verdict and the report can quote the
measured value (a hex code, a percentage of the frame, a contrast ratio).
"""

import colorsys
import difflib
import re
from collections import Counter

import cv2
import numpy as np

from .visual import (
    CMYK_EXTRA_TOLERANCE,
    MissingAsset,
    TextBox,
    contrast_ratio,
    relative_luminance,
    delta_e,
    parse_color,
    rgb_to_lab,
    to_hex,
)

CheckOutcome = tuple[bool, str, list[float]]
SMALL_WIDTH = 240


def _small(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    return cv2.resize(frame, (SMALL_WIDTH, max(1, int(h * SMALL_WIDTH / w))), interpolation=cv2.INTER_NEAREST)


class Palette:
    """Colours from a rule, each with its own tolerance (wider for colours known only in CMYK)."""

    def __init__(self, entries, tolerance: float, cmyk_profile: str | None = None):
        parsed = [parse_color(e, cmyk_profile) for e in entries]
        self.rgb = [rgb for rgb, _, _ in parsed]
        self.labs = np.stack([rgb_to_lab(rgb.reshape(1, 1, 3))[0, 0] for rgb in self.rgb])
        self.tolerances = np.array([tolerance + (CMYK_EXTRA_TOLERANCE if approx else 0) for _, approx, _ in parsed])
        self.approximate = any(approx for _, approx, _ in parsed)
        self.tolerance = tolerance

    def matches(self, frame_lab: np.ndarray) -> np.ndarray:
        """Boolean mask of pixels within tolerance of any palette colour."""
        dist = delta_e(frame_lab[:, :, None, :], self.labs[None, None, :, :])
        return (dist - self.tolerances).min(axis=2) <= 0

    def note(self) -> str:
        return f"ΔE {self.tolerance:g}" + (f", +{CMYK_EXTRA_TOLERANCE:g} for CMYK-derived colours" if self.approximate else "")


def _fmt_times(times: list[float]) -> str:
    return ", ".join(f"{t:g}s" for t in times[:5])


# ---------- colour ----------

def palette_adherence(p: dict, ctx) -> CheckOutcome:
    va = ctx.visual
    palette = Palette(p["palette"], p.get("tolerance", 12), p.get("cmyk_profile"))
    limit = p.get("max_off_palette_share", 0.08)
    worst, worst_i, worst_color = 0.0, 0, ""
    for i, frame in enumerate(va.frames):
        small = _small(frame)
        off = ~palette.matches(rgb_to_lab(small))
        share = float(off.mean())
        if share > worst:
            colors = Counter(to_hex((px // 16) * 16 + 8) for px in small[off].reshape(-1, 3)[::3])
            worst, worst_i, worst_color = share, i, colors.most_common(1)[0][0] if colors else ""
    ok = worst <= limit
    detail = (f"up to {worst:.0%} of a frame is off-palette (limit {limit:.0%}, tolerance {palette.note()})"
              + (f"; most common stray colour about {worst_color}" if not ok and worst_color else ""))
    return ok, detail, [] if ok else [round(va.times[worst_i], 1)]


def banned_hue(p: dict, ctx) -> CheckOutcome:
    va = ctx.visual
    ranges = p.get("hue_ranges", [])
    min_s, min_v, limit = p.get("min_saturation", 0.35), p.get("min_value", 0.2), p.get("max_share", 0.01)
    hits, worst, sample = [], 0.0, ""
    for i, frame in enumerate(va.frames):
        small = _small(frame)
        hsv = cv2.cvtColor(small.astype(np.float32) / 255.0, cv2.COLOR_RGB2HSV)
        hue = hsv[..., 0]
        mask = np.zeros(hue.shape, dtype=bool)
        for lo, hi in ranges:
            mask |= (hue >= lo) & (hue <= hi)
        mask &= (hsv[..., 1] >= min_s) & (hsv[..., 2] >= min_v)
        share = float(mask.mean())
        if share > limit:
            hits.append(va.times[i])
            if share > worst:
                worst = share
                sample = to_hex(np.median(small[mask], axis=0))
    names = ", ".join(f"{lo}-{hi}°" for lo, hi in ranges)
    if hits:
        return False, f"banned hue ({names}) covers up to {worst:.0%} of the frame, for example {sample}", [round(t, 1) for t in hits]
    return True, f"no pixels in the banned hue range ({names}) above {limit:.0%} of any frame", []


def color_coverage(p: dict, ctx) -> CheckOutcome:
    va = ctx.visual
    target = Palette([p["color"]], p.get("tolerance", 15), p.get("cmyk_profile"))
    limit = p["max_share"]
    worst, worst_i = 0.0, 0
    for i, frame in enumerate(va.frames):
        share = float(target.matches(rgb_to_lab(_small(frame))).mean())
        if share > worst:
            worst, worst_i = share, i
    ok = worst <= limit
    return ok, f"{parse_color(p['color'])[2]} covers up to {worst:.0%} of a frame (limit {limit:.0%}, tolerance {target.note()})", [] if ok else [round(va.times[worst_i], 1)]


# ---------- on-screen text ----------

def _text_boxes(ctx, min_conf: float = 0.6) -> dict[int, list[TextBox]]:
    """OCR boxes per sampled frame, without the brand wordmark (which is part of the logo, not on-screen text)."""
    brand = ctx.brand_name.lower()
    out = {}
    for i, boxes in ctx.visual.ocr.items():
        out[i] = [b for b in boxes if b.confidence >= min_conf and b.text.strip().lower() != brand]
    return out


def text_height(p: dict, ctx) -> CheckOutcome:
    va = ctx.visual
    limit = p["min_fraction"] * p.get("tolerance", 0.85)
    bad, smallest = [], 1.0
    for i, boxes in _text_boxes(ctx).items():
        for b in boxes:
            smallest = min(smallest, b.height)
            if b.height < limit:
                bad.append(va.times[i])
    if bad:
        return False, f"text as small as {smallest:.1%} of frame height (minimum {p['min_fraction']:.0%})", sorted(set(round(t, 1) for t in bad))
    return True, f"smallest text is {smallest:.1%} of frame height (minimum {p['min_fraction']:.0%})", []


def _blocks(boxes: list[TextBox]) -> list[list[TextBox]]:
    """Groups OCR lines into text blocks: lines stacked closely and horizontally overlapping belong together."""
    blocks: list[list[TextBox]] = []
    for b in sorted(boxes, key=lambda b: b.y0):
        for block in blocks:
            last = block[-1]
            gap = b.y0 - last.y1
            overlap = min(b.x1, last.x1) - max(b.x0, last.x0)
            if gap < 0.9 * max(b.height, last.height) and overlap > 0.2 * min(b.width, last.width):
                block.append(b)
                break
        else:
            blocks.append([b])
    return blocks


def text_lines(p: dict, ctx) -> CheckOutcome:
    va = ctx.visual
    limit = p["max_lines"]
    bad, most = [], 0
    for i, boxes in _text_boxes(ctx).items():
        for block in _blocks(boxes):
            most = max(most, len(block))
            if len(block) > limit:
                bad.append(va.times[i])
    if bad:
        return False, f"a text block has {most} lines (maximum {limit})", sorted(set(round(t, 1) for t in bad))
    return True, f"no text block has more than {limit} lines (longest: {most})", []


def text_words_per_line(p: dict, ctx) -> CheckOutcome:
    va = ctx.visual
    limit = p["max_words"]
    bad, most = [], 0
    for i, boxes in _text_boxes(ctx).items():
        for b in boxes:
            words = len(b.text.split())
            most = max(most, words)
            if words > limit:
                bad.append(va.times[i])
    if bad:
        return False, f"a line has {most} words (maximum {limit})", sorted(set(round(t, 1) for t in bad))
    return True, f"no line has more than {limit} words (longest: {most})", []


def text_safe_area(p: dict, ctx) -> CheckOutcome:
    va = ctx.visual
    m = p["margin"]
    bad, worst = [], 0.0
    for i, boxes in _text_boxes(ctx).items():
        for b in boxes:
            overshoot = max(m - b.x0, b.x1 - (1 - m), m - b.y0, b.y1 - (1 - m))
            if overshoot > 0.005:
                bad.append(va.times[i])
                worst = max(worst, overshoot)
    if bad:
        return False, f"text reaches {worst:.0%} into the {m:.0%} safe margin", sorted(set(round(t, 1) for t in bad))
    return True, f"all text is inside the {m:.0%} safe margin", []


def _text_colors(frame: np.ndarray, box: TextBox) -> tuple[np.ndarray, np.ndarray] | None:
    """Estimates (background, text) colour of an OCR box: bg is the dominant colour, text the farthest pixels from it."""
    h, w = frame.shape[:2]
    x0, x1 = int(box.x0 * w), int(np.ceil(box.x1 * w))
    y0, y1 = int(box.y0 * h), int(np.ceil(box.y1 * h))
    crop = frame[max(0, y0) : y1, max(0, x0) : x1].reshape(-1, 3)
    if len(crop) < 20:
        return None
    keys = (crop // 8).astype(np.int32)
    packed = keys[:, 0] * 1024 + keys[:, 1] * 32 + keys[:, 2]
    bg = np.median(crop[packed == np.bincount(packed).argmax()], axis=0)
    dist = np.linalg.norm(crop.astype(np.float32) - bg, axis=1)
    if dist.max() < 30:
        return bg, bg
    far = crop[dist >= np.quantile(dist, 0.85)]
    return bg, np.median(far, axis=0)


def text_contrast(p: dict, ctx) -> CheckOutcome:
    va = ctx.visual
    limit = p["min_ratio"]
    bad, lowest, example = [], 99.0, ""
    for i, boxes in _text_boxes(ctx).items():
        for b in boxes:
            colors = _text_colors(va.frames[i], b)
            if colors is None:
                continue
            ratio = contrast_ratio(*colors)
            if ratio < limit:
                bad.append(va.times[i])
                if ratio < lowest:
                    lowest, example = ratio, f'"{b.text}" ({to_hex(colors[1])} on {to_hex(colors[0])})'
    if bad:
        return False, f"contrast {lowest:.1f}:1 for {example}; minimum {limit}:1", sorted(set(round(t, 1) for t in bad))
    return True, f"all text has at least {limit}:1 contrast", []


def text_color(p: dict, ctx) -> CheckOutcome:
    va = ctx.visual
    (on_dark, dark_approx, dark_name), (on_light, light_approx, light_name) = parse_color(p["on_dark"]), parse_color(p["on_light"])
    threshold, tol = p.get("light_background_luminance", 0.6), p.get("tolerance", 25)
    labs = {"dark": rgb_to_lab(on_dark.reshape(1, 1, 3))[0, 0], "light": rgb_to_lab(on_light.reshape(1, 1, 3))[0, 0]}
    tols = {"dark": tol + (CMYK_EXTRA_TOLERANCE if dark_approx else 0), "light": tol + (CMYK_EXTRA_TOLERANCE if light_approx else 0)}
    bad, example = [], ""
    for i, boxes in _text_boxes(ctx).items():
        for b in boxes:
            colors = _text_colors(va.frames[i], b)
            if colors is None or np.allclose(colors[0], colors[1]):
                continue
            bg, fg = colors
            light_bg = relative_luminance(bg) >= threshold
            expected = "light" if light_bg else "dark"
            diff = float(delta_e(rgb_to_lab(fg.reshape(1, 1, 3))[0, 0], labs[expected]))
            if diff > tols[expected]:
                bad.append(va.times[i])
                example = example or f'"{b.text}" is {to_hex(fg)}, expected {to_hex(on_light if light_bg else on_dark)} on {to_hex(bg)}'
    if bad:
        return False, example, sorted(set(round(t, 1) for t in bad))
    return True, f"text colours match {dark_name} on dark and {light_name} on light backgrounds", []


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower().replace("-", " ")).strip()


def text_present(p: dict, ctx) -> CheckOutcome:
    va = ctx.visual
    target, window, sim = _norm(p["text"]), p.get("last_seconds", 3), p.get("similarity", 0.85)
    indices = [i for i in va.ocr if va.times[i] >= va.duration - window] or [max(va.ocr)]
    best = 0.0
    for i in indices:
        boxes = va.ocr[i]
        # a phrase may wrap over several lines, so also compare each whole text block
        candidates = [b.text for b in boxes] + [" ".join(b.text for b in block) for block in _blocks(boxes)]
        for text in candidates:
            best = max(best, difflib.SequenceMatcher(None, target, _norm(text)).ratio())
            if target in _norm(text):
                best = 1.0
    ok = best >= sim
    return ok, f'"{p["text"]}" {"found" if ok else "not found"} in the last {window}s (best text match {best:.0%})', [] if ok else [round(va.times[indices[-1]], 1)]


def banned_words(p: dict, ctx) -> CheckOutcome:
    va = ctx.visual
    words = [_norm(w) for w in p["words"]]
    found: dict[str, list[float]] = {}
    for i, boxes in _text_boxes(ctx, min_conf=0.5).items():
        for b in boxes:
            for w in words:
                if w and w in _norm(b.text):
                    found.setdefault(w, []).append(round(va.times[i], 1))
    if found:
        times = sorted({t for ts in found.values() for t in ts})
        return False, "on-screen text contains: " + ", ".join(f'"{w}"' for w in found), times
    return True, "no banned words in the on-screen text (spoken words are not checked)", []


# ---------- typeface ----------

def font_match(p: dict, ctx) -> CheckOutcome:
    """Compares the letters on screen with the brand font file (see fonts.py for how and for what it cannot do)."""
    from .fonts import compare_line, hires_frame, load_references

    if not ctx.font_paths:
        raise MissingAsset("this check needs the brand's font file; upload it in the Admin tab")
    try:
        brand, control, names, owner, families = load_references(ctx.font_paths)
    except ValueError as exc:
        raise MissingAsset(str(exc))
    va = ctx.visual
    family = p.get("family") or " / ".join(families)
    min_share = p.get("min_match_share", 0.55)
    lines, seen = [], set()
    for i, boxes in _text_boxes(ctx, min_conf=0.85).items():
        for b in boxes:
            key = _norm(b.text)
            if key in seen or b.height < 0.03 or len(key.replace(" ", "")) < 6:
                continue
            seen.add(key)
            frame = hires_frame(va.path, va.times[i], va.info.width, va.info.height)
            h, w = frame.shape[:2]
            crop = frame[int(b.y0 * h) : int(np.ceil(b.y1 * h)), int(b.x0 * w) : int(np.ceil(b.x1 * w))]
            result = compare_line(crop, brand, control, names, owner)
            if result:
                lines.append((va.times[i], b.text, *result))
            if len(lines) >= 24:
                break
    if not lines:
        return True, "no on-screen text was large and clear enough to compare with the font file", []
    bad = [(t, text, share, like) for t, text, _, share, like in lines if share < min_share]
    if bad:
        t, text, share, like = bad[0]
        extra = f"; they look like {like}" if like else ""
        return False, f'"{text}": only {share:.0%} of its letters match {family}{extra}', sorted({round(t, 1) for t, *_ in bad})
    return True, f"letters in {len(lines)} text lines match {family} (checked against {len(names)} other installed fonts)", []


# ---------- logo ----------

def _end_window(p: dict, va) -> tuple[list[int], list[int]]:
    seconds = p.get("end_card_seconds", 2)
    hits = va.logo_hits
    return [i for i in va.body_indices(seconds) if i in hits], [i for i in va.end_card_indices(seconds) if i in hits]


def logo_first_seen(p: dict, ctx) -> CheckOutcome:
    va = ctx.visual
    seen = [va.times[i] for i, hs in sorted(va.logo_hits.items()) if hs]
    limit = p["seconds"]
    if not seen:
        return False, "the logo was never detected", []
    ok = seen[0] <= limit
    return ok, f"logo first appears at {seen[0]:g}s (required within {limit:g}s)", [] if ok else [seen[0]]


def _bugs(hits, max_width):
    return [h for h in hits if h.width <= max_width]


def logo_bug_corner(p: dict, ctx) -> CheckOutcome:
    va = ctx.visual
    body, _ = _end_window(p, va)
    corner, margin, slack = p["corner"], p.get("safe_margin", 0.05), 0.012
    top, right = "top" in corner, "right" in corner
    bad, seen = [], 0
    for i in body:
        for h in _bugs(va.logo_hits[i], p.get("max_width", 0.25)):
            seen += 1
            cx, cy = h.center
            in_corner = (cx > 0.5) == right and (cy < 0.5) == top
            inside = h.x0 >= margin - slack and h.y0 >= margin - slack and h.x1 <= 1 - margin + slack and h.y1 <= 1 - margin + slack
            if not (in_corner and inside):
                where = ("top" if cy < 0.5 else "bottom") + "-" + ("right" if cx > 0.5 else "left")
                bad.append((va.times[i], where, in_corner))
    if not seen:
        return False, "no corner logo bug was detected", []
    if bad:
        where = bad[0][1]
        why = f"found in the {where} corner" if not bad[0][2] else f"outside the {margin:.0%} safe margin"
        return False, f"logo bug {why}; expected {corner}", sorted({round(t, 1) for t, *_ in bad})
    return True, f"logo bug sits in the {corner} corner inside the {margin:.0%} margin", []


def logo_bug_width(p: dict, ctx) -> CheckOutcome:
    va = ctx.visual
    body, _ = _end_window(p, va)
    widths = [(va.times[i], h.width) for i in body for h in _bugs(va.logo_hits[i], p.get("max_width", 0.25))]
    if not widths:
        return False, "no corner logo bug was detected", []
    median = float(np.median([w for _, w in widths]))
    tol = p.get("tolerance", 0.006)
    ok = p["min_fraction"] - tol <= median <= p["max_fraction"] + tol
    return ok, f"logo bug is {median:.1%} of the frame width (allowed {p['min_fraction']:.0%}-{p['max_fraction']:.0%})", [] if ok else [round(widths[0][0], 1)]


def logo_bug_absent_on_end_card(p: dict, ctx) -> CheckOutcome:
    va = ctx.visual
    _, end = _end_window(p, va)
    if not end:
        return True, "no frames in the end-card window", []
    if not any(h.width > p.get("max_width", 0.25) for i in end for h in va.logo_hits[i]):
        return True, "the video has no end card, so there is no bug on it (see the end card rule)", []
    bad = [round(va.times[i], 1) for i in end if _bugs(va.logo_hits[i], p.get("max_width", 0.25))]
    if bad:
        return False, "the corner logo bug is still visible during the end card", bad
    return True, "no corner logo bug during the end card", []


def end_card(p: dict, ctx) -> CheckOutcome:
    va = ctx.visual
    _, end = _end_window(p, va)
    if not end:
        return False, "the video has no frames in its closing window", []
    palette = Palette(p["background_colors"], p.get("tolerance", 12), p.get("cmyk_profile"))
    solid_min = p.get("min_solid_share", 0.7)
    problems, bad_times = [], []
    for i in end:
        solid = float(palette.matches(rgb_to_lab(_small(va.frames[i]))).mean())
        big = [h for h in va.logo_hits[i] if h.width >= p.get("min_logo_width", 0.25) and abs(h.center[0] - 0.5) <= p.get("center_tolerance", 0.12)]
        issues = []
        if solid < solid_min:
            issues.append(f"only {solid:.0%} of the frame is an allowed background colour")
        if not big:
            issues.append("no large centered logo")
        if issues:
            problems.append("; ".join(issues))
            bad_times.append(round(va.times[i], 1))
    if problems:
        return False, f"final {p.get('seconds', 2)}s is not a valid end card: {problems[0]}", bad_times
    return True, f"final {p.get('seconds', 2)}s is a solid {'/'.join(parse_color(c)[2] for c in p['background_colors'])} end card with a large centered logo", []


CHECKS = {
    "palette_adherence": palette_adherence,
    "banned_hue": banned_hue,
    "color_coverage": color_coverage,
    "text_height": text_height,
    "text_lines": text_lines,
    "text_words_per_line": text_words_per_line,
    "text_safe_area": text_safe_area,
    "text_contrast": text_contrast,
    "text_color": text_color,
    "text_present": text_present,
    "banned_words": banned_words,
    "logo_first_seen": logo_first_seen,
    "logo_bug_corner": logo_bug_corner,
    "logo_bug_width": logo_bug_width,
    "logo_bug_absent_on_end_card": logo_bug_absent_on_end_card,
    "end_card": end_card,
    "font_match": font_match,
}
