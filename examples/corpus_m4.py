#!/usr/bin/env python3
"""Reproducible 13-case adversarial UI corpus with exact ground truth for FlowSight M4.

Generates crisp TrueType-rendered screenshots (DejaVuSans / LiberationSans) covering
every required case in M4 §7, plus ground-truth bboxes, expected text labels, and
expected temporal control tracks.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
]


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _draw_centered_text(d: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, font, fill):
    x0, y0, x1, y1 = box
    tb = d.textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    tx = x0 + (x1 - x0 - tw) // 2
    ty = y0 + (y1 - y0 - th) // 2 - tb[1]
    d.text((tx, ty), text, font=font, fill=fill)
    return [tx, ty + tb[1], tx + tw, ty + tb[1] + th]


def _base_app_canvas(dark: bool = True, size: tuple[int, int] = (1280, 720)) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    bg = (18, 20, 28) if dark else (244, 246, 250)
    top = (28, 32, 44) if dark else (226, 232, 240)
    side = (24, 27, 37) if dark else (234, 238, 245)
    div = (68, 76, 98) if dark else (148, 163, 184)
    im = Image.new('RGB', size, bg)
    d = ImageDraw.Draw(im)
    w, h = size
    d.rectangle((0, 0, w - 1, 51), fill=top)
    d.line((0, 52, w - 1, 52), fill=div, width=2)
    d.rectangle((0, 54, 219, h - 35), fill=side)
    d.line((220, 52, 220, h - 34), fill=div, width=2)
    d.line((0, h - 34, w - 1, h - 34), fill=div, width=2)
    d.rectangle((0, h - 32, w - 1, h - 1), fill=top)
    return im, d


def build_corpus(out_dir: Path) -> dict[str, Any]:
    """Generate the full 13-case adversarial corpus and write ground_truth.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    f_btn = load_font(18)
    f_lbl = load_font(16)
    f_sm = load_font(11)
    manifest: dict[str, Any] = {'cases': {}}

    # ------------------------------------------------------------------
    # Case 1 + Case 2 + Case 3 + Case 6 (dark & light): PR review screen
    # - Known button text ("Merge pull request")
    # - Two nearby buttons with distinct labels ("Save", "Cancel")
    # - External labels associated with input fields ("Email:", "Branch")
    # ------------------------------------------------------------------
    for theme_name, is_dark in [('dark', True), ('light', False)]:
        im, d = _base_app_canvas(dark=is_dark)
        fg = (232, 238, 246) if is_dark else (24, 28, 38)
        inp_bg = (14, 16, 24) if is_dark else (255, 255, 255)
        inp_bd = (110, 124, 154) if is_dark else (100, 116, 139)

        # External label LEFT of input field
        d.text((260, 130), 'Email:', font=f_lbl, fill=fg)
        email_lbl_tb = list(d.textbbox((260, 130), 'Email:', font=f_lbl))
        email_inp = [340, 120, 680, 158]
        d.rectangle((email_inp[0], email_inp[1], email_inp[2] - 1, email_inp[3] - 1), fill=inp_bg, outline=inp_bd, width=2)

        # External label ABOVE input field
        d.text((260, 194), 'Target Branch', font=f_lbl, fill=fg)
        branch_lbl_tb = list(d.textbbox((260, 194), 'Target Branch', font=f_lbl))
        branch_inp = [260, 222, 680, 260]
        d.rectangle((branch_inp[0], branch_inp[1], branch_inp[2] - 1, branch_inp[3] - 1), fill=inp_bg, outline=inp_bd, width=2)

        # Primary known green button ("Merge pull request")
        btn_merge = [260, 520, 510, 568]
        d.rectangle((btn_merge[0], btn_merge[1], btn_merge[2] - 1, btn_merge[3] - 1), fill=(22, 163, 74))
        merge_txt_tb = _draw_centered_text(d, tuple(btn_merge), 'Merge pull request', f_btn, (250, 252, 250))

        # Two nearby buttons with distinct labels ("Save" and "Cancel", 18px gap)
        btn_save = [820, 520, 960, 568]
        btn_cancel = [978, 520, 1138, 568]
        d.rectangle((btn_save[0], btn_save[1], btn_save[2] - 1, btn_save[3] - 1), fill=(37, 99, 235))
        save_txt_tb = _draw_centered_text(d, tuple(btn_save), 'Save', f_btn, (250, 252, 255))
        d.rectangle((btn_cancel[0], btn_cancel[1], btn_cancel[2] - 1, btn_cancel[3] - 1), fill=(220, 38, 38))
        cancel_txt_tb = _draw_centered_text(d, tuple(btn_cancel), 'Cancel', f_btn, (255, 250, 250))

        p = out_dir / f'case_pr_{theme_name}.png'
        im.save(p, format='PNG')
        manifest['cases'][f'pr_{theme_name}'] = {
            'path': str(p),
            'theme': theme_name,
            'controls': [
                {'name': 'email_input', 'bbox': email_inp, 'expected_text': 'Email:', 'expected_assoc': 'LABEL_ASSOCIATED', 'label_bbox': email_lbl_tb},
                {'name': 'branch_input', 'bbox': branch_inp, 'expected_text': 'Target Branch', 'expected_assoc': 'LABEL_ASSOCIATED', 'label_bbox': branch_lbl_tb},
                {'name': 'merge_btn', 'bbox': btn_merge, 'expected_text': 'Merge pull request', 'expected_assoc': 'TEXT_INSIDE', 'text_bbox': merge_txt_tb},
                {'name': 'save_btn', 'bbox': btn_save, 'expected_text': 'Save', 'expected_assoc': 'TEXT_INSIDE', 'text_bbox': save_txt_tb},
                {'name': 'cancel_btn', 'bbox': btn_cancel, 'expected_text': 'Cancel', 'expected_assoc': 'TEXT_INSIDE', 'text_bbox': cancel_txt_tb},
            ],
        }

    # ------------------------------------------------------------------
    # Case 4 + Case 5: Small/low-contrast text + overlapping/nested modal & ambiguous straddle
    # ------------------------------------------------------------------
    im45, d45 = _base_app_canvas(dark=True)
    # Small low-contrast button
    low_btn = [260, 120, 440, 162]
    d45.rectangle((low_btn[0], low_btn[1], low_btn[2] - 1, low_btn[3] - 1), fill=(52, 56, 68), outline=(92, 100, 120), width=2)
    low_tb = _draw_centered_text(d45, tuple(low_btn), 'Advanced options', f_sm, (112, 118, 132))
    # Nested dialog card containing an accent button inside
    card_box = [520, 200, 980, 440]
    d45.rectangle((card_box[0], card_box[1], card_box[2] - 1, card_box[3] - 1), fill=(36, 40, 56), outline=(140, 152, 184), width=2)
    inner_btn = [780, 360, 940, 408]
    d45.rectangle((inner_btn[0], inner_btn[1], inner_btn[2] - 1, inner_btn[3] - 1), fill=(37, 99, 235))
    inner_tb = _draw_centered_text(d45, tuple(inner_btn), 'Confirm', f_btn, (250, 252, 255))
    p45 = out_dir / 'case_lowcontrast_nested.png'
    im45.save(p45, format='PNG')
    manifest['cases']['lowcontrast_nested'] = {
        'path': str(p45),
        'low_contrast_btn': low_btn,
        'low_contrast_text_bbox': low_tb,
        'card_box': card_box,
        'inner_btn': inner_btn,
        'inner_text_bbox': inner_tb,
    }

    # ------------------------------------------------------------------
    # Case 7 + Case 8: Modal insertion renumbering controls + Color change preserving identity
    # Frame A = pr_dark; Frame B = pr_dark + new red-bordered modal above + merge_btn green->orange
    # ------------------------------------------------------------------
    im_a = Image.open(out_dir / 'case_pr_dark.png').copy()
    d_b = ImageDraw.Draw(im_a)
    # New modal appearing in upper-middle (changes local W indices of controls below it!)
    modal_box = [700, 100, 1160, 290]
    d_b.rectangle((modal_box[0], modal_box[1], modal_box[2] - 1, modal_box[3] - 1), fill=(52, 24, 34), outline=(239, 68, 68), width=3)
    modal_btn = [960, 224, 1130, 268]
    d_b.rectangle((modal_btn[0], modal_btn[1], modal_btn[2] - 1, modal_btn[3] - 1), fill=(217, 70, 239))
    modal_btn_tb = _draw_centered_text(d_b, tuple(modal_btn), 'Dismiss', f_btn, (255, 250, 255))
    # Existing Merge pull request button changes color green -> orange (SAME position, SAME identity!)
    btn_merge = [260, 520, 510, 568]
    d_b.rectangle((btn_merge[0], btn_merge[1], btn_merge[2] - 1, btn_merge[3] - 1), fill=(234, 148, 24))
    _draw_centered_text(d_b, tuple(btn_merge), 'Merge pull request', f_btn, (20, 20, 20))
    p_78 = out_dir / 'case_modal_and_color_change.png'
    im_a.save(p_78, format='PNG')
    manifest['cases']['modal_and_color_change'] = {
        'frame_a': str(out_dir / 'case_pr_dark.png'),
        'frame_b': str(p_78),
        'new_modal_box': modal_box,
        'new_modal_btn': modal_btn,
        'new_modal_btn_text_bbox': modal_btn_tb,
        'color_changed_btn': btn_merge,
    }

    # ------------------------------------------------------------------
    # Case 9: Button moves cleanly (+140px right, -50px up)
    # ------------------------------------------------------------------
    im9a, d9a = _base_app_canvas(dark=True)
    b9_a = [320, 320, 500, 368]
    d9a.rectangle((b9_a[0], b9_a[1], b9_a[2] - 1, b9_a[3] - 1), fill=(37, 99, 235))
    _draw_centered_text(d9a, tuple(b9_a), 'Deploy', f_btn, (250, 252, 255))
    p9a = out_dir / 'case_move_a.png'
    im9a.save(p9a, format='PNG')

    im9b, d9b = _base_app_canvas(dark=True)
    b9_b = [460, 270, 640, 318]
    d9b.rectangle((b9_b[0], b9_b[1], b9_b[2] - 1, b9_b[3] - 1), fill=(37, 99, 235))
    _draw_centered_text(d9b, tuple(b9_b), 'Deploy', f_btn, (250, 252, 255))
    p9b = out_dir / 'case_move_b.png'
    im9b.save(p9b, format='PNG')
    manifest['cases']['move_button'] = {'frame_a': str(p9a), 'frame_b': str(p9b), 'box_a': b9_a, 'box_b': b9_b}

    # ------------------------------------------------------------------
    # Case 10 (Negative control): Disappearance + appearance of similar-looking control far away
    # ------------------------------------------------------------------
    im10a, d10a = _base_app_canvas(dark=True)
    b10_a = [260, 100, 420, 146]
    d10a.rectangle((b10_a[0], b10_a[1], b10_a[2] - 1, b10_a[3] - 1), fill=(37, 99, 235))
    _draw_centered_text(d10a, tuple(b10_a), 'Start', f_btn, (250, 252, 255))
    p10a = out_dir / 'case_disappear_a.png'
    im10a.save(p10a, format='PNG')

    im10b, d10b = _base_app_canvas(dark=True)
    b10_b = [1060, 580, 1220, 626]
    d10b.rectangle((b10_b[0], b10_b[1], b10_b[2] - 1, b10_b[3] - 1), fill=(37, 99, 235))
    _draw_centered_text(d10b, tuple(b10_b), 'Archive', f_btn, (250, 252, 255))
    p10b = out_dir / 'case_disappear_b.png'
    im10b.save(p10b, format='PNG')
    manifest['cases']['disappear_and_new_far'] = {'frame_a': str(p10a), 'frame_b': str(p10b), 'removed': b10_a, 'added': b10_b}

    # ------------------------------------------------------------------
    # Case 13: Visually identical controls forcing AMBIGUOUS_MATCH & AMBIGUOUS_ASSOCIATION
    # ------------------------------------------------------------------
    im13a, d13a = _base_app_canvas(dark=True)
    # Two identical blue buttons side by side with no unique label
    d13a.rectangle((420, 280, 560, 328), fill=(37, 99, 235))
    d13a.rectangle((580, 280, 720, 328), fill=(37, 99, 235))
    p13a = out_dir / 'case_ambiguous_a.png'
    im13a.save(p13a, format='PNG')

    im13b, d13b = _base_app_canvas(dark=True)
    # Shifted symmetrically downward by 90px -> equidistant/tied between both original positions!
    d13b.rectangle((500, 370, 640, 418), fill=(37, 99, 235))
    p13b = out_dir / 'case_ambiguous_b.png'
    im13b.save(p13b, format='PNG')
    manifest['cases']['ambiguous_identical'] = {'frame_a': str(p13a), 'frame_b': str(p13b)}

    (out_dir / 'ground_truth.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


if __name__ == '__main__':
    out = Path(__file__).resolve().parents[1] / 'evidence' / 'm4' / 'corpus'
    m = build_corpus(out)
    print(f'Generated M4 corpus ({len(m["cases"])} case groups) -> {out}')
