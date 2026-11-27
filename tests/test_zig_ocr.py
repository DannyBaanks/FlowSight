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
    img = Image.new("RGB", (1200, 300), (14, 18, 26))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
    draw.text((480, 44), "danny@host:~/workspace", fill=(220, 226, 235), font=font)
    draw.text((160, 88), "tab-one:~/workspace", fill=(220, 226, 235), font=font)
    draw.text((760, 88), "tab-two:~/workspace", fill=(220, 226, 235), font=font)
    p = tmp_path / "chrome.png"
    img.save(p)
    ctx = flowsight_screen.run_pipeline(str(p), vf="theme,layout,controls,ocr=backend=zig,ui_map")
    rep = flowsight_screen.format_pipeline_report(ctx, max_chars=16000, ocr_limit=150)
    assert "TOP CHROME STRUCTURE" in rep
    assert "2 tabs" in rep

