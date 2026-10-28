"""Video container formats the tool accepts. ffmpeg decodes all of them, so analysis is format-independent."""

import subprocess
import tempfile
from pathlib import Path

# Offered by the upload widget (extensions without the dot).
SUPPORTED_EXTENSIONS = [
    "mp4", "m4v", "mov", "webm", "mkv", "avi", "wmv", "flv", "mpg", "mpeg", "ts", "mts", "m2ts", "3gp", "ogv", "mxf",
]

# Containers the Gemini Files API accepts as they are; anything else is transcoded to MP4 before upload.
GEMINI_NATIVE_SUFFIXES = {".mp4", ".mpeg", ".mpg", ".mov", ".avi", ".flv", ".webm", ".wmv", ".3gp"}


def gemini_ready(video: Path, workdir: Path) -> Path:
    """Returns a file Gemini can ingest: the video itself, or an H.264/AAC MP4 copy for other containers."""
    if video.suffix.lower() in GEMINI_NATIVE_SUFFIXES:
        return video
    out = workdir / f"{video.stem}_gemini.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vf", "scale='min(1280,iw)':-2",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-c:a", "aac", "-movflags", "+faststart", str(out)],
        check=True,
    )
    return out
