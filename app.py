"""Streamlit UI: admins configure brand rules, QA analysts evaluate videos against them."""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from videoqa.evaluate import evaluate
from videoqa.formats import SUPPORTED_EXTENSIONS
from videoqa.extract import extract_rules
from videoqa.llm import GeminiLLM, MockLLM
from videoqa.models import Brand, CheckType, Rule, Severity, Status
from videoqa.store import font_paths, list_brands, load_brand, logo_paths, save_brand, save_font, save_logo
from videoqa.vision import GeminiVideoReviewer

ROOT = Path(__file__).resolve().parent
SAMPLE_PDF = ROOT / "samples" / "guidelines" / "lumora_brand_guidelines.pdf"
SAMPLE_RULES = ROOT / "samples" / "rules" / "lumora_expected_rules.json"
SAMPLE_LOGOS = sorted((ROOT / "samples" / "assets").glob("lumora_logo_*.png"))
SAMPLE_FONTS = sorted((ROOT / "samples" / "assets" / "fonts").glob("*.otf"))
SAMPLE_VIDEOS = sorted((ROOT / "samples" / "videos").glob("*.mp4"))

STATUS_ICON = {Status.passed: "✅", Status.failed: "❌", Status.error: "⚠️", Status.skipped: "⏭️"}

st.set_page_config(page_title="Brand video QA", page_icon="🎬", layout="wide")


def api_key() -> str | None:
    return st.session_state.get("api_key") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def frame_at(video: Path, seconds: float) -> bytes | None:
    res = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-ss", str(seconds), "-i", str(video), "-frames:v", "1",
         "-vf", "scale=480:-1", "-f", "image2pipe", "-vcodec", "png", "-"],
        capture_output=True,
    )
    return res.stdout or None


def rules_to_frame(rules: list[Rule]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "enabled": r.enabled, "id": r.id, "title": r.title, "severity": r.severity.value,
            "type": r.check_type.value, "description": r.description,
            "criterion / params": r.criterion if r.check_type == CheckType.ai else json.dumps({"check": r.check_key, **r.params}),
        }
        for r in rules
    ])


def apply_edits(rules: list[Rule], frame: pd.DataFrame) -> list[Rule]:
    by_id = {r.id: r for r in rules}
    out = []
    for _, row in frame.iterrows():
        rule = by_id[row["id"]].model_copy(deep=True)
        rule.enabled = bool(row["enabled"])
        rule.title = row["title"]
        rule.severity = Severity(row["severity"])
        rule.description = row["description"]
        if rule.check_type == CheckType.ai:
            rule.criterion = row["criterion / params"]
        else:
            params = json.loads(row["criterion / params"])
            rule.check_key = params.pop("check", rule.check_key)
            rule.params = params
        out.append(rule)
    return out


def admin_tab():
    st.subheader("Brand rules")
    if flash := st.session_state.pop("flash", None):
        st.success(flash)
    brands = list_brands()
    col_pick, col_new = st.columns([2, 3])
    with col_pick:
        selected = st.selectbox("Brand", brands, index=None, placeholder="Select a brand")
    with col_new:
        with st.expander("Create or update a brand from a guideline PDF", expanded=not brands):
            name = st.text_input("Brand name", value="Lumora" if not brands else "")
            upload = st.file_uploader("Brand guideline (PDF)", type="pdf")
            st.caption(f"No guideline handy? Use the bundled sample: `{SAMPLE_PDF.relative_to(ROOT)}`")
            use_sample = st.checkbox("Use the sample Lumora guideline")
            offline = st.checkbox(
                "Offline demo: skip the model and load the hand-written sample rules (Lumora sample only)",
                disabled=not use_sample,
            )
            if st.button("Extract rules", type="primary", disabled=not name or not (upload or use_sample)):
                with tempfile.TemporaryDirectory() as tmp:
                    pdf = SAMPLE_PDF if use_sample else Path(tmp) / "guideline.pdf"
                    if upload:
                        pdf = Path(tmp) / "guideline.pdf"
                        pdf.write_bytes(upload.getvalue())
                    try:
                        llm = MockLLM(fixture=SAMPLE_RULES) if offline else GeminiLLM(api_key=api_key())
                        with st.spinner("Reading the guideline and extracting video rules..."):
                            brand = extract_rules(pdf, name, llm)
                        if use_sample:
                            for logo in SAMPLE_LOGOS:
                                save_logo(brand, logo.name, logo.read_bytes())
                            for font_file in SAMPLE_FONTS:
                                save_font(brand, font_file.name, font_file.read_bytes())
                        save_brand(brand)
                        st.session_state["flash"] = f"Saved {len(brand.rules)} rules for {brand.name}. Select it above to review."
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Extraction failed: {exc}")

    if not selected:
        return
    brand = load_brand(selected)
    logos = logo_paths(brand)
    with st.expander(f"Brand logo files ({len(logos)})", expanded=not logos):
        if not logos:
            st.warning("No logo uploaded: the logo checks (reveal time, bug corner and size, end card) will be skipped.")
        else:
            st.image([str(p) for p in logos], width=140, caption=[p.name for p in logos])
        st.caption("Upload the logo as transparent PNGs. Add one per version if the logo changes colour on light and dark backgrounds.")
        new_logos = st.file_uploader("Logo images", type="png", accept_multiple_files=True, key=f"logos_{brand.name}")
        if st.button("Save logos", disabled=not new_logos):
            for f in new_logos:
                save_logo(brand, f.name, f.getvalue())
            save_brand(brand)
            st.rerun()
    fonts = font_paths(brand)
    with st.expander(f"Brand font files ({len(fonts)})", expanded=False):
        if fonts:
            st.write(", ".join(p.name for p in fonts))
        else:
            st.info("Optional. With no font file the typeface rule is judged by Gemini; with one it is measured "
                    "by comparing the letters on screen with the font. Upload every weight the guideline allows "
                    "(for example Regular and Bold).")
        new_fonts = st.file_uploader("Font files (TTF or OTF)", type=["ttf", "otf"], accept_multiple_files=True, key=f"fonts_{brand.name}")
        if st.button("Save fonts", disabled=not new_fonts):
            for f in new_fonts:
                save_font(brand, f.name, f.getvalue())
            save_brand(brand)
            st.rerun()
    st.caption(f"Source: {brand.source_file or 'n/a'} · {len(brand.rules)} rules. Edit cells, then save.")
    edited = st.data_editor(
        rules_to_frame(brand.rules), hide_index=True, width="stretch", key=f"rules_{brand.name}",
        disabled=["id", "type"],
        column_config={
            "enabled": st.column_config.CheckboxColumn(width="small"),
            "severity": st.column_config.SelectboxColumn(options=[s.value for s in Severity], width="small"),
            "type": st.column_config.TextColumn(width="small"),
        },
    )
    if st.button("Save rule changes"):
        try:
            brand.rules = apply_edits(brand.rules, edited)
            save_brand(brand)
            st.success("Saved.")
        except Exception as exc:
            st.error(f"Could not save: {exc}")


def report_view(report, video_path: Path):
    judged = [r for r in report.results if r.status in (Status.passed, Status.failed)]
    skipped = [r for r in report.results if r.status == Status.skipped]
    errors = [r for r in report.results if r.status == Status.error]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Compliance score", f"{report.score:.0f}/100")
    m2.metric("Passed", sum(r.status == Status.passed for r in judged))
    m3.metric("Failed", len(report.failures))
    m4.metric("Not checked", len(skipped) + len(errors))
    if skipped:
        st.info(f"{len(skipped)} AI rules were not checked. Enable AI checks with a Gemini key for a full review; "
                "the score covers only the checks that ran.")
    if errors:
        st.warning(f"{len(errors)} checks could not be completed: {errors[0].detail}")

    if report.failures:
        st.markdown("#### Issues to fix")
        for r in sorted(report.failures, key=lambda r: ["critical", "major", "minor"].index(r.severity.value)):
            with st.container(border=True):
                when = f" · at {', '.join(f'{t:g}s' for t in r.timestamps)}" if r.timestamps else ""
                st.markdown(f"**{r.title}** · `{r.severity.value}`{when}")
                st.write(r.detail)
                frames = [frame_at(video_path, t) for t in r.timestamps[:3]]
                images = [f for f in frames if f]
                if images:
                    st.image(images, caption=[f"{t:g}s" for t in r.timestamps[: len(images)]])

    st.markdown("#### All checks")
    st.dataframe(
        pd.DataFrame([
            {"": STATUS_ICON[r.status], "rule": r.title, "severity": r.severity.value, "how": r.check_type.value,
             "result": r.detail, "at (s)": ", ".join(f"{t:g}" for t in r.timestamps)}
            for r in report.results
        ]),
        hide_index=True, width="stretch",
    )
    st.download_button("Download report (JSON)", report.model_dump_json(indent=2),
                       file_name=f"{Path(report.video).stem}_report.json", mime="application/json")


def qa_tab():
    st.subheader("Evaluate a video")
    brands = list_brands()
    if not brands:
        st.info("No brands yet. An admin needs to create one in the Admin tab first.")
        return
    brand_name = st.selectbox("Brand", brands, key="qa_brand")
    upload = st.file_uploader("Video asset", type=SUPPORTED_EXTENSIONS)
    sample = st.selectbox("...or try a sample", ["(none)"] + [p.name for p in SAMPLE_VIDEOS])
    use_ai = st.checkbox("Run AI checks with Gemini", value=bool(api_key()),
                         help="Needs a Gemini API key. Technical checks always run.")
    if use_ai and not api_key():
        st.warning("Add a Gemini API key in the sidebar to run AI checks.")

    if st.button("Evaluate", type="primary", disabled=not (upload or sample != "(none)")):
        with tempfile.TemporaryDirectory() as tmp:
            if upload:
                video = Path(tmp) / upload.name
                video.write_bytes(upload.getvalue())
            else:
                video = ROOT / "samples" / "videos" / sample
            try:
                reviewer = GeminiVideoReviewer(api_key=api_key()) if use_ai and api_key() else None
                with st.spinner("Running checks..."):
                    report = evaluate(video, load_brand(brand_name), reviewer)
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")
                return
            st.video(str(video))
            report_view(report, video)


def main():
    st.title("🎬 Brand video QA")
    with st.sidebar:
        st.header("Settings")
        st.text_input("Gemini API key", type="password", key="api_key",
                      help="Free key from aistudio.google.com/apikey. Kept only in this browser session. "
                           "You can also set GEMINI_API_KEY in the environment.")
        st.caption("Key detected" if api_key() else "No key set: AI extraction and AI checks are unavailable.")
    admin, qa = st.tabs(["Admin: brand rules", "QA: evaluate a video"])
    with admin:
        admin_tab()
    with qa:
        qa_tab()


main()
