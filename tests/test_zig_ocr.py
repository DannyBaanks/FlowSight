"""Tests for FlowSight Zig Shape OCR backend (`backend='zig'`)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import flowsight_ocr
import flowsight_screen


def _has_zig_ocr() -> bool:
    bin_path, atlas_path, _ = flowsight_ocr.detect_zig_ocr_engine()
    return bool(bin_path and atlas_path)


@pytest.mark.skipif(not _has_zig_ocr(), reason="flowsight-ocr-zig binary or atlas.fsoa not built")
def test_zig_ocr_reads_spanish_punctuation_and_spacing():
    img = Image.new("RGB", (800, 160), (12, 18, 26))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 15)
    font_ui = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    draw.text((20, 24), "Esperando respuesta del modelo · 4s · paso 8", fill=(230, 236, 245), font=font)
    draw.text((20, 68), "¡Hola! ¿En qué te ayudo hoy?", fill=(220, 230, 240), font=font)
    draw.rectangle([660, 20, 760, 56], fill=(8, 199, 250))
    draw.text((678, 29), "CODER", fill=(12, 18, 26), font=font_ui)

    rgb = np.asarray(img, dtype=np.uint8)
    res = flowsight_ocr.run_ocr(rgb, 0, 0, "synthetic_zig", backend="zig")
    assert res["status"] == flowsight_ocr.OCR_OBSERVED
    assert res["engine"] == "flowsight-ocr-zig/0.1"
    joined = " | ".join(it["text"] for it in res["items"])
    assert "Esperando respuesta del modelo" in joined
    assert "CODER" in joined
    assert "¡" in joined and "¿" in joined and "qué" in joined


@pytest.mark.skipif(not _has_zig_ocr(), reason="flowsight-ocr-zig binary or atlas.fsoa not built")
def test_zig_ocr_empty_image_returns_ocr_empty():
    blank = np.full((120, 240, 3), 24, dtype=np.uint8)
    res = flowsight_ocr.run_ocr(blank, 0, 0, "blank_zig", backend="zig")
    assert res["status"] == flowsight_ocr.OCR_EMPTY
    assert res["items"] == []


@pytest.mark.skipif(not _has_zig_ocr(), reason="flowsight-ocr-zig binary or atlas.fsoa not built")
def test_oneshot_report_detects_top_chrome_tabs(tmp_path):
    # Non-Q13 geometry (title y=12, tab strip y=48) with genuine visual tab-strip evidence
    # (distinct active-tab fill + continuous horizontal divider below tab strip).
    img = Image.new("RGB", (960, 300), (20, 24, 32))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    draw.rectangle([0, 0, 960, 38], fill=(16, 20, 28))
    draw.text((360, 12), "Mi Navegador", fill=(220, 226, 235), font=font)
    draw.rectangle([0, 38, 960, 72], fill=(26, 30, 40))
    draw.rectangle([40, 40, 260, 72], fill=(58, 66, 84))  # active tab distinct fill
    draw.text((64, 48), "Inicio del proyecto", fill=(242, 246, 252), font=font)
    draw.text((320, 48), "Documentación API", fill=(180, 188, 202), font=font)
    draw.line([0, 72, 960, 72], fill=(95, 108, 130), width=1)
    p = tmp_path / "chrome.png"
    img.save(p)
    ctx = flowsight_screen.run_pipeline(str(p), vf="theme,layout,controls,ocr=backend=zig,ui_map")
    rep = flowsight_screen.format_pipeline_report(ctx, max_chars=16000, ocr_limit=150)
    assert "TOP CHROME STRUCTURE (INFERRED" in rep
    assert "DEMONSTRATED" not in rep.split("TOP CHROME STRUCTURE")[1].splitlines()[0]
    assert "2 tabs" in rep


@pytest.mark.skipif(not _has_zig_ocr(), reason="flowsight-ocr-zig binary or atlas.fsoa not built")
def test_top_chrome_rejects_toolbar_buttons_and_form_labels(tmp_path):
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)

    # Negative 1: Toolbar with two buttons ("Guardar", "Exportar") — must NOT be labeled tabs
    img_btn = Image.new("RGB", (900, 300), (24, 28, 36))
    d = ImageDraw.Draw(img_btn)
    d.text((320, 34), "Editor de documento", fill=(230, 235, 242), font=font)
    d.rectangle([120, 74, 230, 104], fill=(48, 60, 82), outline=(80, 96, 120))
    d.text((142, 81), "Guardar", fill=(235, 240, 248), font=font)
    d.rectangle([340, 74, 460, 104], fill=(48, 60, 82), outline=(80, 96, 120))
    d.text((362, 81), "Exportar", fill=(235, 240, 248), font=font)
    p_btn = tmp_path / "buttons.png"
    img_btn.save(p_btn)
    rep_btn = flowsight_screen.format_pipeline_report(
        flowsight_screen.run_pipeline(str(p_btn), vf="ocr=backend=zig"), max_chars=8000, ocr_limit=50
    )
    assert "TOP CHROME STRUCTURE" not in rep_btn

    # Negative 2: Form labels on uniform background — must NOT be labeled tabs
    img_frm = Image.new("RGB", (900, 300), (245, 246, 248))
    d = ImageDraw.Draw(img_frm)
    d.text((340, 34), "Nuevo registro", fill=(25, 28, 34), font=font)
    d.text((100, 82), "Nombre completo", fill=(40, 45, 55), font=font)
    d.text((420, 82), "Correo electrónico", fill=(40, 45, 55), font=font)
    p_frm = tmp_path / "form.png"
    img_frm.save(p_frm)
    rep_frm = flowsight_screen.format_pipeline_report(
        flowsight_screen.run_pipeline(str(p_frm), vf="ocr=backend=zig"), max_chars=8000, ocr_limit=50
    )
    assert "TOP CHROME STRUCTURE" not in rep_frm


@pytest.mark.skipif(not _has_zig_ocr(), reason="flowsight-ocr-zig binary or atlas.fsoa not built")
def test_zig_ocr_rejects_pure_noise_and_checkerboard():
    rng = np.random.default_rng(42)
    noise = rng.integers(0, 256, size=(540, 960, 3), dtype=np.uint8)
    res_n = flowsight_ocr._run_zig_ocr(noise, 0, 0, "noise", "spa+eng", 60.0, 10.0)
    assert res_n["status"] == flowsight_ocr.OCR_EMPTY
    assert res_n["items"] == []

    yy, xx = np.indices((300, 400))
    checker = np.stack([(((xx // 4 + yy // 4) % 2) * 255).astype(np.uint8)] * 3, axis=-1)
    res_c = flowsight_ocr._run_zig_ocr(checker, 0, 0, "checker", "spa+eng", 60.0, 10.0)
    assert res_c["status"] == flowsight_ocr.OCR_EMPTY
    assert res_c["items"] == []


@pytest.mark.skipif(not _has_zig_ocr(), reason="flowsight-ocr-zig binary or atlas.fsoa not built")
def test_zig_ocr_rejects_smooth_textures_but_reads_text_on_them():
    """Smooth textures (blurred photos/wallpapers) have a dominant background bin, so the
    coherence gate alone lets them through; the edge-sharpness gate must reject them."""
    rng = np.random.default_rng(7)
    for scale in (8, 24):
        coarse = rng.integers(0, 256, size=(540 // scale + 2, 960 // scale + 2, 3), dtype=np.uint8)
        smooth = np.asarray(Image.fromarray(coarse).resize((960, 540), Image.BICUBIC))
        res = flowsight_ocr.run_ocr(smooth, 0, 0, f"smooth{scale}", backend="zig")
        assert res["items"] == [], f"scale={scale} hallucinated {[it['text'] for it in res['items']][:8]}"

    coarse = np.random.default_rng(7).integers(0, 80, size=(9, 26, 3), dtype=np.uint8)
    img = Image.fromarray(np.asarray(Image.fromarray(coarse).resize((600, 100), Image.BICUBIC)))
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
    ImageDraw.Draw(img).text((20, 40), "error: file not found", fill=(250, 250, 250), font=font)
    res = flowsight_ocr.run_ocr(np.asarray(img), 0, 0, "text_on_texture", backend="zig")
    assert any("file not found" in it["text"] for it in res["items"]), [it["text"] for it in res["items"]]


@pytest.mark.skipif(not _has_zig_ocr(), reason="flowsight-ocr-zig binary or atlas.fsoa not built")
def test_zig_ocr_preserves_hyphens_underscores_and_stacked_lines():
    img = Image.new("RGB", (640, 140), (18, 22, 30))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 13)
    draw.text((20, 20), "2026-11-27 11:38", fill=(230, 236, 245), font=font)
    draw.text((20, 38), "isycode_mcp_execute", fill=(230, 236, 245), font=font)
    draw.text((20, 56), "Buscar archivos...", fill=(230, 236, 245), font=font)

    rgb = np.asarray(img, dtype=np.uint8)
    res = flowsight_ocr.run_ocr(rgb, 0, 0, "stacked_zig", backend="zig")
    assert res["status"] == flowsight_ocr.OCR_OBSERVED
    texts = [it["text"] for it in res["items"]]
    assert len(texts) == 3, f"expected 3 separate lines, got {texts}"
    assert "2026-11-27 11:38" in texts[0]
    assert "sycode_mcp_execute" in texts[1]
    assert "Buscar arch" in texts[2] and "..." in texts[2]
    assert "S" not in texts[2] and "C" not in texts[2] and "V" not in texts[2]


@pytest.mark.skipif(not _has_zig_ocr(), reason="flowsight-ocr-zig binary or atlas.fsoa not built")
def test_zig_ocr_serve_matches_oneshot_and_recovers(monkeypatch):
    img = Image.new("RGB", (640, 120), (14, 18, 26))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
    draw.text((20, 24), "FlowSight serve mode 2026", fill=(230, 236, 245), font=font)
    draw.text((20, 60), "¡Hola! paso 8", fill=(220, 230, 240), font=font)
    rgb = np.asarray(img, dtype=np.uint8)

    # Persistent --serve mode (bypass high-level OCR cache to exercise server directly)
    res_serve = flowsight_ocr._run_zig_ocr(rgb, 0, 0, "serve_frame", "spa+eng", 60.0, 10.0)
    assert res_serve["status"] == flowsight_ocr.OCR_OBSERVED
    assert res_serve["preprocessing"] == "serve_rgb_bt601"
    proc = flowsight_ocr._ZIG_OCR_SERVER._proc
    assert proc is not None and proc.poll() is None
    old_pid = proc.pid

    # Kill worker and verify transparent auto-recovery
    proc.kill()
    proc.wait(timeout=2.0)
    res_rec = flowsight_ocr._run_zig_ocr(rgb, 0, 0, "serve_recover", "spa+eng", 60.0, 10.0)
    assert res_rec["status"] == flowsight_ocr.OCR_OBSERVED
    assert res_rec["preprocessing"] == "serve_rgb_bt601"
    assert flowsight_ocr._ZIG_OCR_SERVER._proc is not None
    assert flowsight_ocr._ZIG_OCR_SERVER._proc.pid != old_pid

    # Exact parity against one-shot PGM fallback
    monkeypatch.setenv("FLOWSIGHT_ZIG_OCR_NO_SERVE", "1")
    res_one = flowsight_ocr._run_zig_ocr(rgb, 0, 0, "oneshot_frame", "spa+eng", 60.0, 10.0)
    assert res_one["preprocessing"] == "grayscale_pgm_p5"
    assert [it["text"] for it in res_serve["items"]] == [it["text"] for it in res_one["items"]]
    assert [it["text"] for it in res_rec["items"]] == [it["text"] for it in res_one["items"]]

    # Clean shutdown leaves no child process
    flowsight_ocr.shutdown_zig_ocr_server()
    assert flowsight_ocr._ZIG_OCR_SERVER._proc is None



