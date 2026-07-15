"""
Contact sheet generator.

Composes a single image showing a query header and a horizontal row of result
images. Used by run_eval.py to produce eyeball-check outputs.

Pillow only — no matplotlib, no scipy. The goal is a quick visual sanity check,
not a publication-quality figure.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


THUMB_W = 224
THUMB_H = 280
HEADER_H = 60
SCORE_H = 30
PADDING = 8
FONT_SIZE = 13


def _load_font(size: int) -> ImageFont.ImageFont:
    """Try to load a readable system font; fall back to PIL default."""
    candidates = [
        "arial.ttf", "Arial.ttf",
        "DejaVuSans.ttf",
        "LiberationSans-Regular.ttf",
        "Helvetica.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def make_contact_sheet(
    query: str,
    results: list[dict],
    output_path: str,
    thumb_w: int = THUMB_W,
    thumb_h: int = THUMB_H,
) -> None:
    """
    Render and save a contact sheet for one query.

    Args:
        query:       The raw query string (shown as header).
        results:     List of dicts with keys: image_path, final_score,
                     stage1_score, attribute_score, setting_score.
        output_path: Where to save the .png file.
        thumb_w/h:   Thumbnail dimensions for each result image.
    """
    n = len(results)
    if n == 0:
        return

    font = _load_font(FONT_SIZE)
    small_font = _load_font(FONT_SIZE - 2)

    sheet_w = n * (thumb_w + PADDING) + PADDING
    sheet_h = HEADER_H + thumb_h + SCORE_H + PADDING * 2

    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(30, 30, 35))
    draw = ImageDraw.Draw(sheet)

    # Header: query text, wrapped to fit.
    header_text = f'Query: "{query}"'
    wrapped = textwrap.fill(header_text, width=max(30, sheet_w // 8))
    draw.text((PADDING, PADDING), wrapped, fill=(220, 220, 220), font=font)

    # Thumbnails + score overlay.
    for i, result in enumerate(results):
        x_offset = PADDING + i * (thumb_w + PADDING)
        y_offset = HEADER_H

        img_path = result.get("image_path", "")

        # Load and thumbnail the result image.
        if img_path and os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert("RGB")
                img.thumbnail((thumb_w, thumb_h), Image.LANCZOS)
                # Paste centered on a dark background in case thumbnail is smaller.
                bg = Image.new("RGB", (thumb_w, thumb_h), (50, 50, 55))
                px = (thumb_w - img.width) // 2
                py = (thumb_h - img.height) // 2
                bg.paste(img, (px, py))
                img = bg
            except Exception:
                img = _error_thumbnail(thumb_w, thumb_h)
        else:
            img = _error_thumbnail(thumb_w, thumb_h)

        sheet.paste(img, (x_offset, y_offset))

        # Rank badge (top-left corner of thumbnail).
        rank_label = f"#{i + 1}"
        draw.rectangle(
            [x_offset, y_offset, x_offset + 28, y_offset + 18],
            fill=(60, 60, 200),
        )
        draw.text((x_offset + 3, y_offset + 2), rank_label, fill=(255, 255, 255), font=small_font)

        # Score line below thumbnail.
        score = result.get("final_score", 0.0)
        s1 = result.get("stage1_score", 0.0)
        attr = result.get("attribute_score", 0.0)
        score_text = f"{score:.3f} (s1={s1:.2f} a={attr:.2f})"
        draw.text(
            (x_offset, y_offset + thumb_h + 3),
            score_text,
            fill=(180, 210, 180),
            font=small_font,
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(str(output_path))
    print(f"Contact sheet saved: {output_path}")


def _error_thumbnail(w: int, h: int) -> Image.Image:
    """Return a placeholder thumbnail for missing or unreadable images."""
    img = Image.new("RGB", (w, h), (80, 40, 40))
    draw = ImageDraw.Draw(img)
    draw.text((10, h // 2 - 8), "Not found", fill=(200, 100, 100))
    return img
