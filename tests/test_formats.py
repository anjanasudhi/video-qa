import subprocess
from pathlib import Path

import pytest

from videoqa.formats import SUPPORTED_EXTENSIONS, gemini_ready
from videoqa.probe import probe
from videoqa.technical import CHECKS, VideoContext

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "videos" / "lumora_compliant_9x16_6s.mp4"


def remux(src: Path, dst: Path, *extra: str):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), *extra, str(dst)], check=True)


@pytest.mark.parametrize("suffix, codec", [(".mkv", ["-c", "copy"]), (".avi", ["-c:v", "mpeg4", "-c:a", "mp3"]), (".ts", ["-c", "copy"])])
def test_other_containers_are_analysed_like_mp4(tmp_path, suffix, codec):
    """Temporary re-wraps of an existing sample; no extra sample files are kept."""
    other = tmp_path / f"clip{suffix}"
    remux(SAMPLE, other, *codec)
    info = probe(other)
    assert (info.width, info.height) == (1080, 1920) and info.has_audio
    ctx = VideoContext.open(other)
    ok, detail, _ = CHECKS["aspect_ratio"]({"allowed": ["9:16"]}, ctx)
    assert ok, detail
    ok, detail, _ = CHECKS["duration"]({"allowed_seconds": [6], "tolerance": 0.5}, ctx)
    assert ok, detail


def test_gemini_gets_mp4_for_containers_it_does_not_take(tmp_path):
    assert gemini_ready(SAMPLE, tmp_path) == SAMPLE
    mkv = tmp_path / "clip.mkv"
    remux(SAMPLE, mkv, "-c", "copy")
    converted = gemini_ready(mkv, tmp_path)
    assert converted.suffix == ".mp4" and converted != mkv and probe(converted).duration == pytest.approx(6, abs=0.5)


def test_upload_list_covers_common_formats():
    assert {"mp4", "mov", "mkv", "avi", "webm"} <= set(SUPPORTED_EXTENSIONS)
