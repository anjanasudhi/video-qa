"""Generates the sample Lumora videos: two compliant ones and two with deliberate violations.

Frames are drawn with Pillow and piped to ffmpeg; the soundtrack is a synthetic arpeggio.
Ground truth for each video is written to samples/videos/manifest.json so the evaluator
can be scored against it later.
"""

import json
import subprocess
import tempfile
import wave
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from brand import FONT_BOLD, FONT_OFF_BRAND, FONT_REGULAR, PALETTE, TAGLINE, hex_to_rgb
from logo import render_logo

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "samples" / "videos"

VIOLET = PALETTE["Lumora Violet"]
CORAL = PALETTE["Sunrise Coral"]
MIDNIGHT = PALETTE["Midnight"]
WHITE = PALETTE["Cloud White"]
OFF_GREEN = "#2ECC71"
LIGHT_GRAY = "#D9D9D9"


@dataclass
class Spec:
    name: str
    w: int
    h: int
    fps: int
    duration: float
    scenes: list  # (start, end, bg_hex)
    texts: list  # (start, end, lines, font_path, color_hex, size_frac, y_frac, x_frac)
    captions: list = field(default_factory=list)  # (start, end, text)
    bug: dict | None = None  # corner, width_frac, start, end
    endcard: dict | None = None  # start, bg, tagline
    flash: tuple | None = None  # start, end, hz
    gain_db: float = 0.0  # applied after normalising to -16 LUFS
    violations: list = field(default_factory=list)
    description: str = ""


@lru_cache(maxsize=None)
def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


@lru_cache(maxsize=None)
def logo(width: int, on_dark: bool) -> Image.Image:
    base = render_logo(240, on_dark=on_dark)
    return base.resize((width, round(base.height * width / base.width)), Image.LANCZOS)


def luminance(hex_value: str) -> float:
    r, g, b = hex_to_rgb(hex_value)
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255


def active(items, t):
    return [it for it in items if it[0] <= t < it[1]]


def render_frame(spec: Spec, t: float) -> Image.Image:
    w, h = spec.w, spec.h
    scene_bg = next((bg for s, e, bg in spec.scenes if s <= t < e), spec.scenes[-1][2])
    in_end = spec.endcard is not None and t >= spec.endcard["start"]
    bg = spec.endcard["bg"] if in_end else scene_bg

    if spec.flash and spec.flash[0] <= t < spec.flash[1]:
        on = int((t - spec.flash[0]) * spec.flash[2] * 2) % 2 == 0
        return Image.new("RGB", (w, h), (255, 255, 255) if on else (0, 0, 0))

    img = Image.new("RGB", (w, h), hex_to_rgb(bg))
    draw = ImageDraw.Draw(img)

    if not in_end:
        # Rising "sun": a small coral disc drifting up from the lower edge of the frame.
        scene_start, scene_end, _ = next(s for s in spec.scenes if s[0] <= t < s[1])
        progress = (t - scene_start) / max(scene_end - scene_start, 0.01)
        radius = int(h * 0.08)
        cx, cy = int(w * 0.88), int(h * (1.05 - 0.25 * progress))
        sun_color = CORAL if luminance(bg) < 0.9 else VIOLET
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=hex_to_rgb(sun_color))

    for start, end, lines, font_path, color, size_frac, y_frac, x_frac in active(spec.texts, t):
        f = font(font_path, int(size_frac * h))
        line_h = int(size_frac * h * 1.25)
        y0 = int(y_frac * h) - line_h * (len(lines) - 1) // 2
        for i, line in enumerate(lines):
            draw.text((int(x_frac * w), y0 + i * line_h), line, font=f, fill=hex_to_rgb(color), anchor="mm")

    for start, end, text in active(spec.captions, t):
        f = font(FONT_REGULAR, int(0.04 * h))
        box = draw.textbbox((int(w / 2), int(0.8 * h)), text, font=f, anchor="mm")
        pad = int(0.012 * h)
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(overlay).rounded_rectangle(
            (box[0] - pad * 2, box[1] - pad, box[2] + pad * 2, box[3] + pad),
            radius=pad, fill=hex_to_rgb(MIDNIGHT) + (170,),
        )
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        ImageDraw.Draw(img).text((int(w / 2), int(0.8 * h)), text, font=f, fill=hex_to_rgb(WHITE), anchor="mm")

    if in_end and spec.endcard:
        mark = logo(int(w * 0.4), on_dark=True)
        img.paste(mark, ((w - mark.width) // 2, int(h * 0.42) - mark.height // 2), mark)
        if spec.endcard.get("tagline", True):
            ImageDraw.Draw(img).text(
                (w // 2, int(h * 0.42) + mark.height), TAGLINE,
                font=font(FONT_REGULAR, int(0.045 * h)), fill=hex_to_rgb(WHITE), anchor="mm",
            )

    bug = spec.bug
    if bug and bug["start"] <= t < bug["end"] and (bug.get("persist_on_end") or not in_end):
        bw = int(bug["width_frac"] * w)
        mark = logo(bw, on_dark=luminance(bg) < 0.6)
        mx, my = int(0.05 * w), int(0.05 * h)
        x = w - mx - mark.width if "right" in bug["corner"] else mx
        y = my if "top" in bug["corner"] else h - my - mark.height
        img.paste(mark, (x, y), mark)

    return img


def make_audio(path: Path, duration: float, gain_db: float, seed: int = 7):
    """Synthetic warm arpeggio, loudness-normalised to -16 LUFS, then boosted by gain_db."""
    sr = 48000
    n = int(sr * duration)
    rng = np.random.default_rng(seed)
    chords = [(261.63, 329.63, 392.0, 523.25), (220.0, 261.63, 329.63, 440.0),
              (174.61, 220.0, 261.63, 349.23), (196.0, 246.94, 293.66, 392.0)]
    audio = np.zeros((n, 2))
    step = 0.5
    k = 0
    t_note = 0.0
    while t_note < duration:
        freq = chords[int(t_note // 2) % 4][k % 4]
        i0 = int(t_note * sr)
        length = min(int(0.9 * sr), n - i0)
        tt = np.arange(length) / sr
        env = np.minimum(tt / 0.02, 1.0) * np.exp(-tt * 3.5)
        tone = env * (np.sin(2 * np.pi * freq * tt) + 0.3 * np.sin(2 * np.pi * 2 * freq * tt))
        pan = 0.5 + 0.2 * rng.uniform(-1, 1)
        audio[i0 : i0 + length, 0] += tone * (1 - pan)
        audio[i0 : i0 + length, 1] += tone * pan
        t_note += step
        k += 1
    audio *= 0.3 / max(np.abs(audio).max(), 1e-9)

    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.wav"
        with wave.open(str(raw), "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes((audio * 32767).astype("<i2").tobytes())
        filters = "loudnorm=I=-16:TP=-1.5:LRA=7"
        if gain_db:
            filters += f",volume={gain_db}dB"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-af", filters,
             "-ar", str(sr), "-c:a", "pcm_s16le", str(path)],
            check=True,
        )


def render_video(spec: Spec):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{spec.name}.mp4"
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        make_audio(wav, spec.duration, spec.gain_db)
        proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{spec.w}x{spec.h}", "-r", str(spec.fps), "-i", "-",
             "-i", str(wav),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20",
             "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
             "-t", str(spec.duration), "-movflags", "+faststart", str(out)],
            stdin=subprocess.PIPE,
        )
        for i in range(int(round(spec.duration * spec.fps))):
            proc.stdin.write(render_frame(spec, i / spec.fps).tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed for {spec.name}")
    print(f"wrote {out}")
    return out


def measure_loudness(path: Path) -> dict:
    res = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af", "ebur128=peak=true", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    summary = res.stderr.split("Summary:")[-1]
    out = {}
    for line in summary.splitlines():
        line = line.strip()
        if line.startswith("I:"):
            out["integrated_lufs"] = float(line.split()[1])
        elif line.startswith("Peak:"):
            out["true_peak_dbtp"] = float(line.split()[1])
    return out


def specs() -> list[Spec]:
    inter_b, inter_r = FONT_BOLD, FONT_REGULAR

    compliant_wide = Spec(
        name="lumora_compliant_16x9_15s", w=1920, h=1080, fps=30, duration=15.0,
        scenes=[(0, 4, MIDNIGHT), (4, 8, VIOLET), (8, 13, MIDNIGHT)],
        texts=[
            (0.3, 4, ["Light that follows your day"], inter_b, WHITE, 0.07, 0.45, 0.5),
            (4.2, 8, ["Warm at sunrise."], inter_b, WHITE, 0.07, 0.45, 0.5),
            (8.2, 13, ["Focused for work.", "Calm at night."], inter_b, WHITE, 0.07, 0.45, 0.5),
        ],
        captions=[
            (0.5, 4, "Meet Lumora, the lamp that follows your day."),
            (4.2, 8, "It wakes you with warm light at sunrise."),
            (8.2, 13, "Focused for work, calm at night."),
        ],
        bug={"corner": "top-right", "width_frac": 0.10, "start": 0.5, "end": 13.0},
        endcard={"start": 13.0, "bg": VIOLET, "tagline": True},
        description="Fully compliant 16:9 ad.",
    )

    compliant_tall = Spec(
        name="lumora_compliant_9x16_6s", w=1080, h=1920, fps=30, duration=6.0,
        scenes=[(0, 2, MIDNIGHT), (2, 4, VIOLET)],
        texts=[
            (0.3, 2, ["Light that", "follows your day"], inter_b, WHITE, 0.05, 0.45, 0.5),
            (2.2, 4, ["Calm at night."], inter_b, WHITE, 0.05, 0.45, 0.5),
        ],
        captions=[(0.5, 2, "Meet Lumora."), (2.2, 4, "Calm light at night.")],
        bug={"corner": "top-right", "width_frac": 0.115, "start": 0.5, "end": 4.0},
        endcard={"start": 4.0, "bg": MIDNIGHT, "tagline": True},
        description="Fully compliant 9:16 six-second cutdown.",
    )

    bad_brand = Spec(
        name="lumora_violations_brand_16x9_15s", w=1920, h=1080, fps=30, duration=15.0,
        scenes=[(0, 4, MIDNIGHT), (4, 9, OFF_GREEN), (9, 13, VIOLET)],
        texts=[
            (0.3, 4, ["Light that follows your day"], FONT_OFF_BRAND, WHITE, 0.07, 0.45, 0.5),
            (4.2, 9, ["Our revolutionary smart lamp quickly adjusts", "its warm light to match every part", "of your busy day automatically"],
             FONT_OFF_BRAND, WHITE, 0.07, 0.45, 0.5),
            (9.2, 13, ["Calm at night."], FONT_OFF_BRAND, WHITE, 0.07, 0.45, 0.5),
        ],
        captions=[],
        bug={"corner": "bottom-left", "width_frac": 0.15, "start": 0.5, "end": 15.0, "persist_on_end": True},
        endcard={"start": 13.0, "bg": VIOLET, "tagline": True},
        violations=["logo_bug_position", "logo_bug_size", "logo_bug_end_card", "banned_color",
                    "font", "text_lines", "text_words_per_line", "safe_area", "captions", "tone_of_voice"],
        description="Right format and audio, but wrong logo bug corner/size, serif font, green scene, text overflow, no captions.",
    )

    bad_tech = Spec(
        name="lumora_violations_tech_16x9_22s", w=1280, h=720, fps=24, duration=22.0,
        scenes=[(0, 5, WHITE), (5, 12, MIDNIGHT), (12, 22, WHITE)],
        texts=[
            (0.3, 5, ["Smart lamps for your home"], inter_r, LIGHT_GRAY, 0.07, 0.45, 0.5),
            (5.2, 12, ["Light that follows your day"], inter_b, WHITE, 0.07, 0.45, 0.5),
            (12.2, 22, ["Calm at night."], inter_r, LIGHT_GRAY, 0.07, 0.45, 0.5),
        ],
        captions=[],
        bug={"corner": "top-right", "width_frac": 0.10, "start": 5.0, "end": 22.0},
        endcard=None,
        flash=(8.0, 10.0, 8),
        gain_db=14.0,
        violations=["resolution", "frame_rate", "duration", "brand_reveal", "end_card", "captions",
                    "loudness", "clipping", "flashing", "contrast"],
        description="720p/24fps, 22s, late logo reveal, no end card, no captions, clipped loud audio, strobing, low-contrast text.",
    )

    return [compliant_wide, compliant_tall, bad_brand, bad_tech]


def main():
    manifest = {}
    for spec in specs():
        path = render_video(spec)
        manifest[path.name] = {
            "brand": "Lumora",
            "compliant": not spec.violations,
            "expected_violations": spec.violations,
            "description": spec.description,
            "loudness": measure_loudness(path),
        }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
