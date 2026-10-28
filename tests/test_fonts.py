from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from videoqa.fonts import compare_line, control_fonts, load_references

FONTS = Path(__file__).resolve().parent.parent / "samples" / "assets" / "fonts"
SERIF = Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf")
SANS = Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf")


def text_crop(font_path: Path, text: str = "Light that follows your day", size: int = 80) -> np.ndarray:
    font = ImageFont.truetype(str(font_path), size)
    img = Image.new("RGB", (int(font.getlength(text)) + 80, size * 2), (20, 20, 43))
    ImageDraw.Draw(img).text((40, size // 3), text, font=font, fill=(255, 255, 255))
    arr = np.array(img)
    ys, xs = np.where(arr.mean(axis=2) > 100)
    return arr[ys.min() - 4 : ys.max() + 5, xs.min() - 4 : xs.max() + 5]


@pytest.fixture(scope="module")
def refs():
    try:
        return load_references(sorted(FONTS.glob("*.otf")))
    except ValueError:
        pytest.skip("not enough other fonts installed to compare against")


def test_brand_font_text_matches(refs):
    brand, control, names, owner, _ = refs
    letters, share, _ = compare_line(text_crop(FONTS / "Inter-Bold.otf"), brand, control, names, owner)
    assert letters >= 20 and share >= 0.7


@pytest.mark.parametrize("other", [SERIF, SANS])
def test_other_typefaces_are_caught_and_named(refs, other):
    if not other.exists():
        pytest.skip(f"{other.name} not installed")
    brand, control, names, owner, _ = refs
    _, share, looks_like = compare_line(text_crop(other), brand, control, names, owner)
    assert share < 0.55 and looks_like.split()[0] in other.stem


def test_brand_family_is_never_a_control_font(refs):
    families = refs[4]
    assert "Inter" in families
    assert not any(Path(p).name.startswith("Inter-") for p in control_fonts(tuple(families)))


def test_tiny_text_is_not_judged(refs):
    brand, control, names, owner, _ = refs
    assert compare_line(np.full((10, 40, 3), 200, np.uint8), brand, control, names, owner) is None
