"""Draws the Lumora logo with Pillow so the samples need no external assets."""

from PIL import Image, ImageDraw, ImageFont

from brand import FONT_BOLD, PALETTE, hex_to_rgb


def render_logo(height: int, on_dark: bool = True) -> Image.Image:
    """Return the horizontal logo (sun mark + wordmark) as an RGBA image."""
    violet = hex_to_rgb(PALETTE["Lumora Violet"])
    coral = hex_to_rgb(PALETTE["Sunrise Coral"])
    text_color = hex_to_rgb(PALETTE["Cloud White"] if on_dark else PALETTE["Midnight"])

    font = ImageFont.truetype(FONT_BOLD, int(height * 0.62))
    text = "lumora"
    text_w = int(font.getlength(text))
    gap = int(height * 0.22)
    width = height + gap + text_w

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Mark: violet disc with a coral rising sun.
    draw.ellipse((0, 0, height - 1, height - 1), fill=violet + (255,))
    sun_r = int(height * 0.22)
    cx, cy = height // 2, int(height * 0.62)
    draw.pieslice((cx - sun_r, cy - sun_r, cx + sun_r, cy + sun_r), 180, 360, fill=coral + (255,))
    line_w = max(2, height // 28)
    draw.line((height * 0.2, cy, height * 0.8, cy), fill=(255, 255, 255, 255), width=line_w)

    bbox = font.getbbox(text)
    text_y = (height - (bbox[3] - bbox[1])) // 2 - bbox[1]
    draw.text((height + gap, text_y), text, font=font, fill=text_color + (255,))
    return img


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "lumora_logo.png"
    render_logo(240).save(out)
    print(f"wrote {out}")
