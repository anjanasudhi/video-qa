from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent


def test_admin_offline_extract_then_evaluate_technical_only(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDEOQA_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception
    at.checkbox[0].check().run()  # use the sample guideline
    at.checkbox[1].check().run()  # offline demo rules
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert any("Saved 27 rules" in s.value for s in at.success)
    assert (tmp_path / "lumora.json").exists()

    # QA tab: pick a sample video and run technical checks only
    at.selectbox[-1].select("lumora_violations_tech_16x9_22s.mp4").run()
    next(b for b in at.button if b.label == "Evaluate").click().run()
    assert not at.exception, at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Failed"] == "12"
