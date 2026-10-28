"""Thin wrappers over ffprobe / ffmpeg for the measurements technical checks need."""

import json
import re
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: float
    has_audio: bool


def probe(path: str | Path) -> VideoInfo:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    data = json.loads(out)
    video = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    if video is None:
        raise ValueError("File has no video stream")
    fps = float(Fraction(video.get("avg_frame_rate") or video["r_frame_rate"]))
    duration = float(data["format"].get("duration") or video.get("duration") or 0)
    return VideoInfo(
        width=int(video["width"]), height=int(video["height"]), fps=fps, duration=duration,
        has_audio=any(s["codec_type"] == "audio" for s in data["streams"]),
    )


def measure_audio(path: str | Path) -> dict[str, float]:
    """Integrated loudness (LUFS) and true peak (dBTP) via ffmpeg's EBU R128 scanner."""
    res = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-vn", "-af", "ebur128=peak=true", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    summary = res.stderr.split("Summary:")[-1]
    out = {}
    if m := re.search(r"I:\s+(-?[\d.]+) LUFS", summary):
        out["lufs"] = float(m.group(1))
    if m := re.search(r"Peak:\s+(-?[\d.]+) dBFS", summary):
        out["true_peak"] = float(m.group(1))
    return out


def mean_luminance(path: str | Path, fps: float) -> np.ndarray:
    """Mean brightness (0-255) of every frame, from a tiny grayscale downscale."""
    w, h = 32, 18
    proc = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(path), "-vf", f"fps={fps},scale={w}:{h}", "-pix_fmt", "gray",
         "-f", "rawvideo", "-"],
        capture_output=True, check=True,
    )
    frames = np.frombuffer(proc.stdout, dtype=np.uint8)
    return frames[: len(frames) // (w * h) * (w * h)].reshape(-1, w * h).mean(axis=1)
