"""Builds the sample Lumora brand guideline PDF.

The guideline mixes general brand rules with a dedicated video section, so the
extraction step has to find the video-relevant rules across the whole document.
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from brand import BRAND_NAME, PALETTE, TAGLINE
from logo import render_logo

ROOT = Path(__file__).resolve().parent.parent
OUT_PDF = ROOT / "samples" / "guidelines" / "lumora_brand_guidelines.pdf"
ASSETS = ROOT / "samples" / "assets"

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], textColor=colors.HexColor(PALETTE["Lumora Violet"]))
H2 = ParagraphStyle("H2", parent=styles["Heading2"], textColor=colors.HexColor(PALETTE["Midnight"]))
BODY = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=10.5, leading=15)
BULLET = ParagraphStyle("Bullet", parent=BODY, leftIndent=12, bulletIndent=2)


def p(text):
    return Paragraph(text, BODY)


def bullets(items):
    return [Paragraph(item, BULLET, bulletText="•") for item in items]


def color_table():
    rows = [["Swatch", "Name", "HEX", "Role"]]
    roles = {
        "Lumora Violet": "Primary. Backgrounds, end cards, key shapes.",
        "Sunrise Coral": "Accent. Highlights and calls to action only.",
        "Midnight": "Text on light backgrounds, dark backgrounds.",
        "Cloud White": "Text on brand colors, light backgrounds.",
    }
    for name, hex_value in PALETTE.items():
        rows.append(["", name, hex_value, roles[name]])
    table = Table(rows, colWidths=[18 * mm, 35 * mm, 25 * mm, 90 * mm])
    style = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
    ]
    for i, hex_value in enumerate(PALETTE.values(), start=1):
        style.append(("BACKGROUND", (0, i), (0, i), colors.HexColor(hex_value)))
    table.setStyle(TableStyle(style))
    return table


def build():
    ASSETS.mkdir(parents=True, exist_ok=True)
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    logo_light = ASSETS / "lumora_logo_on_light.png"
    logo_dark = ASSETS / "lumora_logo_on_dark.png"
    render_logo(240, on_dark=False).save(logo_light)
    render_logo(240, on_dark=True).save(logo_dark)

    story = []

    # Cover
    story += [Spacer(1, 60 * mm), Image(str(logo_light), width=110 * mm, height=110 * mm * 240 / 760)]
    story += [Spacer(1, 10 * mm), Paragraph(f"{BRAND_NAME} Brand Guidelines", H1)]
    story += [p(f"Version 2.1 &middot; October 2026 &middot; <i>{TAGLINE}</i>")]
    story += [Spacer(1, 6 * mm), p("<font size=8>Fictional brand created as sample data for the video-qa prototype.</font>")]
    story.append(PageBreak())

    # 1. Brand story
    story += [Paragraph("1. Who we are", H1)]
    story += [p(
        f"{BRAND_NAME} makes smart lamps that follow the rhythm of your day: warm light at sunrise, "
        "focused light for work, calm light at night. Our brand should feel optimistic, warm and quietly "
        "intelligent, never cold or overly technical."
    )]
    story += [Paragraph("Brand personality", H2)]
    story += bullets(["Warm, not saccharine.", "Clever, not complicated.", "Calm, not sleepy."])

    # 2. Logo
    story += [Paragraph("2. Logo", H1)]
    story += [p(
        "The logo combines the sun mark with the lowercase <b>lumora</b> wordmark. Always use the supplied "
        "artwork. Use the white wordmark on Lumora Violet, Midnight or photography, and the Midnight wordmark "
        "on white."
    )]
    story += [Paragraph("Clear space and size", H2)]
    story += bullets([
        "Keep clear space around the logo equal to the height of the sun mark.",
        "Minimum size in print: 25 mm wide. On screen: never smaller than 120 px wide or 8% of the canvas width, whichever is larger.",
    ])
    story += [Paragraph("Logo misuse", H2)]
    story += bullets([
        "Do not stretch, rotate, recolor or add effects (shadows, outlines, glows) to the logo.",
        "Do not place the logo on busy backgrounds where it loses contrast.",
        "Do not lock up the logo with partner or competitor logos without brand team approval.",
    ])
    story.append(PageBreak())

    # 3. Color
    story += [Paragraph("3. Color", H1)]
    story += [p(
        "Our palette is led by Lumora Violet. Sunrise Coral is an accent: it should cover no more than "
        "about 20% of any layout. Green tones are reserved by our main competitor and must never be used "
        "in graphics, backgrounds or type."
    )]
    story += [Spacer(1, 4 * mm), color_table()]

    # 4. Typography
    story += [Paragraph("4. Typography", H1)]
    story += [p(
        "Inter is our only typeface, across print, web, app and video. Use Inter Bold for headlines and Inter "
        "Regular for body copy and captions. Serif typefaces are never used. Headlines are written in "
        "sentence case."
    )]

    # 5. Voice
    story += [Paragraph("5. Voice and tone", H1)]
    story += bullets([
        "Speak like a thoughtful friend: short sentences, second person (\"you\"), no jargon.",
        "Avoid hype words such as \"revolutionary\", \"insane\" or \"game-changing\".",
        "Always write the brand name as Lumora (capital L), never LUMORA or lumora in running text.",
    ])
    story.append(PageBreak())

    # 6. Video and motion
    story += [Paragraph("6. Video and motion", H1)]
    story += [p(
        "These rules apply to every video asset, including ads, social cutdowns, product demos and "
        "website loops. Agencies must check deliverables against this section before submission."
    )]

    story += [Paragraph("6.1 Formats and technical specs", H2)]
    story += bullets([
        "Accepted aspect ratios: 16:9 (1920&times;1080) for web and TV, 9:16 (1080&times;1920) for Stories and Reels, 1:1 (1080&times;1080) for feeds.",
        "Minimum resolution is 1080 px on the short side. 720p or lower will be rejected.",
        "Frame rate must be 25 or 30 fps. Do not deliver 24 fps or variable frame rate files.",
        "Deliver H.264 MP4 with AAC stereo audio.",
    ])

    story += [Paragraph("6.2 Duration", H2)]
    story += bullets([
        "Approved lengths are 6, 15 and 30 seconds, with a tolerance of &plusmn;0.5 seconds.",
        "Website loops may be up to 60 seconds but must loop seamlessly.",
    ])

    story += [Paragraph("6.3 Brand reveal", H2)]
    story += bullets([
        "The Lumora logo or brand name must be visible within the first 2 seconds.",
    ])

    story += [Paragraph("6.4 Logo bug", H2)]
    story += bullets([
        "A persistent logo bug sits in the <b>top-right corner</b>, inside the 5% safe margin, from the brand reveal until the end card.",
        "The bug must be 8&ndash;12% of the frame width. Never place it in another corner.",
        "Remove the bug during the end card so the logo does not appear twice.",
    ])

    story += [Paragraph("6.5 End card", H2)]
    story += bullets([
        "Every video ends with a 2-second end card: the full logo centered on a solid Lumora Violet or Midnight background.",
        f"The tagline \"{TAGLINE}\" appears beneath the logo in Inter Regular, Cloud White.",
    ])

    story += [Paragraph("6.6 On-screen text", H2)]
    story += bullets([
        "On-screen text uses Inter only (see section 4). Text height at least 4% of the frame height.",
        "No more than 2 lines, and no more than 8 words per line.",
        "Keep all text inside the 10% title-safe area.",
        "Cloud White on brand colors or footage; Midnight on white.",
    ])

    story += [Paragraph("6.7 Captions", H2)]
    story += bullets([
        "All web and social videos must carry burned-in captions for every spoken line and key message, because most viewers watch with sound off.",
        "Captions sit in the lower third, centered, in Cloud White Inter Regular on a semi-transparent Midnight box.",
    ])

    story += [Paragraph("6.8 Color in motion graphics", H2)]
    story += bullets([
        "Backgrounds, shapes, lower thirds and transitions use only the brand palette (section 3).",
        "No green backgrounds or graphics at any point.",
    ])

    story += [Paragraph("6.9 Audio", H2)]
    story += bullets([
        "Integrated loudness: &minus;16 LUFS &plusmn;1.5 LU for web and social. True peak must not exceed &minus;1 dBTP. No clipping or distortion.",
        "Music should be warm and upbeat. No lyrics underneath voiceover.",
        "Voiceover is warm and conversational, matching section 5.",
    ])

    story += [Paragraph("6.10 Accessibility and safety", H2)]
    story += bullets([
        "No flashing content more than 3 times per second.",
        "Maintain at least 4.5:1 contrast between text and background.",
    ])

    story += [Paragraph("7. Photography", H1)]
    story += [p(
        "Use natural, warm light and real homes. Avoid stock-looking studio setups and cold blue color grading. "
        "People should look relaxed and candid."
    )]

    doc = SimpleDocTemplate(
        str(OUT_PDF), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm, title=f"{BRAND_NAME} Brand Guidelines",
    )
    doc.build(story)
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    build()
