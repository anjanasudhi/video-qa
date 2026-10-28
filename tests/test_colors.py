import numpy as np
import pytest

from videoqa.visual import CMYK_EXTRA_TOLERANCE, cmyk_to_rgb, contrast_ratio, parse_color, rgb_to_lab
from videoqa.visual_checks import Palette


def test_cmyk_conversion_extremes():
    assert cmyk_to_rgb(0, 0, 0, 0).tolist() == [255, 255, 255]
    assert cmyk_to_rgb(0, 0, 0, 100).tolist() == [0, 0, 0]
    assert cmyk_to_rgb(100, 0, 0, 0).tolist() == [0, 255, 255]


def test_parse_color_prefers_exact_values_over_cmyk():
    rgb, approx, label = parse_color({"name": "Violet", "hex": "#5B2EFF", "cmyk": [72, 81, 0, 0]})
    assert rgb.tolist() == [91, 46, 255] and not approx and label == "Violet"
    rgb, approx, _ = parse_color({"cmyk": [0, 60, 60, 0]})
    assert approx and rgb.tolist() == [255, 102, 102]
    assert parse_color({"rgb": [1, 2, 3]})[0].tolist() == [1, 2, 3]
    with pytest.raises(ValueError):
        parse_color({"name": "nothing"})


def test_cmyk_only_palette_gets_a_wider_tolerance():
    # the screen colour is a little off the naive CMYK conversion, as it would be in real footage
    frame = np.full((2, 2, 3), (255, 118, 100), dtype=np.uint8)
    lab = rgb_to_lab(frame)
    exact = Palette(["#FF6666"], tolerance=5)
    cmyk = Palette([{"cmyk": [0, 60, 60, 0]}], tolerance=5)
    assert not exact.matches(lab).any()
    assert cmyk.matches(lab).all() and cmyk.approximate
    assert f"+{CMYK_EXTRA_TOLERANCE:g}" in cmyk.note()


def test_contrast_ratio_matches_wcag():
    assert contrast_ratio((0, 0, 0), (255, 255, 255)) == pytest.approx(21.0)
    assert contrast_ratio((119, 119, 119), (255, 255, 255)) == pytest.approx(4.48, abs=0.02)
