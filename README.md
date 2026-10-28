# video-qa

Brand video QA prototype. An **admin** uploads a brand guideline and reviews the video rules extracted from it; a **QA analyst** uploads a video and gets a compliance report against those rules.

How it works:

1. **Extract rules.** The guideline PDF is read and a Gemini model returns structured rules, each one either a *technical* check or an *AI* check (a yes/no question about the video).
2. **Technical checks** run locally with ffprobe/ffmpeg: aspect ratio, resolution, frame rate, duration, integrated loudness, true peak, flashing.
3. **AI checks** (logo placement, end card, fonts, captions, palette, text safe area, tone...) are reviewed in one request by Gemini, which watches the video and listens to its audio.
4. The report has a severity-weighted score, pass/fail per rule, timestamps and evidence frames.

## Setup

Requires Python 3.11+ and `ffmpeg`/`ffprobe` on the PATH.

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...   # free key: https://aistudio.google.com/apikey
streamlit run app.py
```

Without a key you can still use the UI: load the sample rules with the **offline demo** option in the Admin tab, and run technical checks only in the QA tab. The key can also be pasted into the sidebar (kept in the browser session only). The model defaults to `gemini-2.5-flash`; override with `GEMINI_MODEL`. The Gemini free tier may use submitted content to improve Google's models, so don't send confidential assets through it.

Command line:

```bash
python -m videoqa extract samples/guidelines/lumora_brand_guidelines.pdf --brand Lumora
python -m videoqa evaluate samples/videos/lumora_violations_brand_16x9_15s.mp4 --brand Lumora [--no-ai]
```

Brand rules are saved as JSON in `data/brands/` (override with `VIDEOQA_DATA_DIR`).

## Sample data

`samples/` holds a fictional brand, Lumora: a guideline PDF with a video section, four generated test videos (two compliant, two with deliberate violations) and `samples/videos/manifest.json` with the expected violations for each. `scripts/make_guideline.py` and `scripts/make_videos.py` regenerate them (they need `reportlab`, plus the Inter font installed).

## Tests

```bash
pytest
```

The technical checks are verified against the sample videos' ground truth. Gemini calls are covered with mocks only; the live Gemini paths (rule extraction and video review) have not been run against the real API yet.

## Layout

- `videoqa/extract.py`: PDF to rules; `videoqa/models.py`: rule and report schema
- `videoqa/technical.py`, `videoqa/probe.py`: deterministic checks
- `videoqa/vision.py`: Gemini video review (swap in another reviewer by implementing `review()`)
- `videoqa/evaluate.py`: runs the rules and builds the report
- `app.py`: Streamlit UI
