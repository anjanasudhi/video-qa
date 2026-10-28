# video-qa

Brand video QA prototype. An **admin** uploads a brand guideline and reviews the video rules extracted from it; a **QA analyst** uploads a video and gets a compliance report against those rules.

How it works:

1. **Extract rules.** The guideline PDF is read and a Gemini model returns structured rules. Each rule is either a *measured* check or an *AI* check (a yes/no question about the video).
2. **Measured checks** are deterministic: the same video always gives the same verdict, and the report quotes the measured value.
   - Specs via ffprobe/ffmpeg: aspect ratio, resolution, frame rate, duration, integrated loudness, true peak, flashing.
   - Colour: palette adherence, banned hues (for example no green), and how much of the frame a colour covers. Colours can be given as hex, RGB or CMYK; CMYK-only colours are converted to sRGB with the standard formula (or an ICC profile) and matched with a wider tolerance, because that conversion is approximate.
   - On-screen text via OCR: height, lines, words per line, safe area, text colour, WCAG contrast, required wording (such as the end-card tagline) and banned words.
   - Typeface (optional, needs the brand font files): every letter on screen is compared with the brand font and with up to 40 other installed fonts. A line passes when most letters are closer to the brand font, and a failure names the font it looks like. Upload every allowed weight (for example Regular and Bold). It reliably catches serif, mono and clearly different sans faces; two near-identical designs can still be confused.
   - Logo placement via template matching against the brand's logo file: first appearance, corner, size, absence on the end card, and the end card itself.
3. **AI checks** (typeface, caption style, music, tone, anything that needs judgement) are reviewed in one request by Gemini, which watches the video and listens to its audio. The typeface check is measured when the admin uploads the brand font file (see below); otherwise Gemini judges it.
4. The report has a severity-weighted score, pass/fail per rule, timestamps and evidence frames.

## Setup

Requires Python 3.11+ and `ffmpeg`/`ffprobe` on the PATH.

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...   # free key: https://aistudio.google.com/apikey
streamlit run app.py
```

Without a key you can still use the UI: load the sample rules with the **offline demo** option in the Admin tab, and run the measured checks only in the QA tab (the AI checks are skipped and the report says so). The key can also be pasted into the sidebar (kept in the browser session only). The model defaults to `gemini-3.5-flash`; override with `GEMINI_MODEL`. The Gemini free tier may use submitted content to improve Google's models, so don't send confidential assets through it.

Command line:

```bash
python -m videoqa extract samples/guidelines/lumora_brand_guidelines.pdf --brand Lumora
python -m videoqa evaluate samples/videos/lumora_violations_brand_16x9_15s.mp4 --brand Lumora [--no-ai]
```

Brand rules are saved as JSON in `data/brands/` (override with `VIDEOQA_DATA_DIR`). Upload the brand's logo (transparent PNGs) in the Admin tab; without it the logo checks are skipped rather than guessed (or judged by Gemini when AI review is on). Font files (TTF/OTF) are optional and work the same way.

**Speed and limits.** Evaluating a video takes roughly 1-2 seconds of processing per second of video for the measured checks (about 35 s for a 15 s 1080p clip on 4 cores), plus about 15-20 s for the Gemini review. At most 150 frames are sampled, so very long videos are analysed more coarsely. The uploader accepts MP4, M4V, MOV, WEBM, MKV, AVI, WMV, FLV, MPG/MPEG, TS/MTS/M2TS, 3GP, OGV and MXF (anything ffmpeg can decode; the list is in `videoqa/formats.py`) up to Streamlit's 200 MB default. Containers Gemini does not take directly are converted to a 720p-class MP4 copy for the AI review only; the measured checks always read your original file. Logo detection assumes the logo sits on flat graphics, not busy footage.

## Sample data

`samples/` holds a fictional brand, Lumora: a guideline PDF with a video section, four generated test videos (two compliant, two with deliberate violations) and `samples/videos/manifest.json` with the expected violations for each. `scripts/make_guideline.py` and `scripts/make_videos.py` regenerate them (they need `reportlab`, plus the Inter font installed).

## Tests

```bash
pytest   # about 4 minutes: it analyses all four sample videos
```

The measured checks are verified against the sample videos' ground truth (every planted violation is found and nothing else is flagged). Gemini calls are mocked in the tests; the live extraction and video review were run by hand against the sample guideline and videos.

## Layout

- `videoqa/extract.py`: PDF to rules; `videoqa/models.py`: rule and report schema
- `videoqa/technical.py`, `videoqa/probe.py`: spec, audio and flashing checks
- `videoqa/visual.py`, `videoqa/visual_checks.py`, `videoqa/fonts.py`: colour, OCR text, logo and typeface checks
- `videoqa/vision.py`: Gemini video review (swap in another reviewer by implementing `review()`)
- `videoqa/evaluate.py`: runs the rules and builds the report
- `app.py`: Streamlit UI
