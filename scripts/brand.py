"""Constants for "Lumora", the fictional brand used by the sample data."""

BRAND_NAME = "Lumora"
TAGLINE = "Light up every moment."

PALETTE = {
    "Lumora Violet": "#5B2EFF",
    "Sunrise Coral": "#FF6B5B",
    "Midnight": "#14142B",
    "Cloud White": "#FFFFFF",
}

FONT_DIR = "/usr/share/fonts/opentype/inter"
FONT_REGULAR = f"{FONT_DIR}/Inter-Regular.otf"
FONT_BOLD = f"{FONT_DIR}/Inter-Bold.otf"
# Used only by the off-brand test video.
FONT_OFF_BRAND = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))
