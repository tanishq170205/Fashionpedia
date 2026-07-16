"""
Dominant color extraction for garment crops.

Design notes:
- We store the actual measured RGB value, not just the quantized color name.
  At retrieval time the reranker compares query color words against stored RGB
  using Euclidean distance, so "burgundy" in a query can match a crop measured
  at [128, 20, 30] even though our palette rounds that to "maroon". Storing
  only the name would discard the precision needed for that comparison.

- K-means is used instead of a trained color classifier because there is no
  labeled per-pixel color data available for Fashionpedia val_test2020, and the
  dominant-cluster approach works well on solid or near-solid garments, which
  are the majority of this corpus. The main failure mode is patterned or
  multi-color garments (e.g., plaid, stripes): K-means picks the modal cluster,
  which may not match how a person would describe the garment's color.

- The center-crop before resizing discards the outer 10% of the bounding box
  on each side to reduce background bleed-in from loose detector boxes. This is
  a cheap but effective heuristic; a semantic segmentation mask would do better.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


# Reference palette: human color name → (R, G, B).
# This is intentionally small — 16 entries — because the goal is fast nearest-
# neighbor lookup, not a perceptually complete color space. The names are chosen
# to cover the most common fashion color descriptors without overlap.
_REFERENCE_PALETTE: dict[str, Tuple[int, int, int]] = {
    "black":   (20,  20,  20),
    "white":   (240, 240, 240),
    "gray":    (128, 128, 128),
    "red":     (200,  30,  30),
    "maroon":  (128,   0,   0),
    "orange":  (230, 120,  30),
    "yellow":  (230, 210,  30),
    "olive":   (128, 128,   0),
    "green":   ( 30, 160,  50),
    "teal":    (  0, 128, 128),
    "navy":    (  0,   0, 128),
    "blue":    ( 30,  90, 200),
    "purple":  (128,   0, 128),
    "pink":    (230, 130, 160),
    "brown":   (130,  70,  30),
    "beige":   (220, 200, 160),
}


def extract_dominant_color(
    image: Image.Image,
    n_clusters: int = 5,
    center_crop_fraction: float = 0.8,
) -> dict:
    """
    Return the dominant color of a PIL image (assumed to be a garment crop).

    Returns a dict with:
        color_rgb:  [R, G, B] as Python ints — the actual measured cluster center
        color_name: nearest human label from the reference palette (for display/debug)

    Raises ValueError if the image has fewer pixels than n_clusters after resizing.
    """
    # Center-crop to reduce background contamination from loose bounding boxes.
    w, h = image.size
    margin_w = int(w * (1 - center_crop_fraction) / 2)
    margin_h = int(h * (1 - center_crop_fraction) / 2)
    if margin_w > 0 or margin_h > 0:
        image = image.crop((margin_w, margin_h, w - margin_w, h - margin_h))

    # Resize to a fixed small size before K-means; the cluster result is stable
    # above ~32px per side and the speedup from reducing pixel count is large.
    image = image.convert("RGB").resize((64, 64), Image.LANCZOS)
    pixels = np.array(image).reshape(-1, 3).astype(np.float32)

    if len(pixels) < n_clusters:
        raise ValueError(
            f"Crop has only {len(pixels)} pixels, cannot fit {n_clusters} clusters."
        )

    kmeans = KMeans(n_clusters=n_clusters, n_init=3, random_state=42)
    labels = kmeans.fit_predict(pixels)

    # Pick the cluster whose centroid represents the most pixels.
    counts = np.bincount(labels)
    dominant_idx = int(np.argmax(counts))
    dominant_rgb = kmeans.cluster_centers_[dominant_idx].astype(int).tolist()

    color_name = _map_to_color_name(dominant_rgb)

    return {
        "color_rgb": dominant_rgb,
        "color_name": color_name,
    }


def _map_to_color_name(rgb: list[int]) -> str:
    """Map an [R, G, B] list to the nearest name in the reference palette."""
    best_name = "unknown"
    best_dist = float("inf")
    r, g, b = rgb
    for name, (pr, pg, pb) in _REFERENCE_PALETTE.items():
        dist = math.sqrt((r - pr) ** 2 + (g - pg) ** 2 + (b - pb) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name


def color_name_to_rgb(color_word: str) -> Tuple[int, int, int] | None:
    """
    Convert a color word (potentially a query synonym like "burgundy", "cobalt",
    "ecru") to an approximate RGB triple.

    Lookup order:
    1. Direct hit in our reference palette (exact match, case-insensitive).
    2. Synonym table covering common fashion color terms.
    3. Python webcolors CSS4 names as a catch-all.
    4. Return None if all lookups fail (caller must handle gracefully).
    """
    word = color_word.strip().lower()

    # Normalize compound color names that LLMs commonly produce.
    # Maps two-word phrases to the single canonical word in our tables.
    _COMPOUNDS: dict[str, str] = {
        "navy blue":    "navy",
        "dark blue":    "navy",
        "light blue":   "sky",
        "sky blue":     "sky",
        "royal blue":   "royal",
        "cobalt blue":  "cobalt",
        "dark green":   "forest",
        "light green":  "mint",
        "olive green":  "olive",
        "dark red":     "maroon",
        "light pink":   "blush",
        "hot pink":     "hot pink",
        "dark brown":   "brown",
        "dark gray":    "charcoal",
        "dark grey":    "charcoal",
        "light gray":   "silver",
        "light grey":   "silver",
        "off white":    "off-white",
        "off-white":    "off-white",
        "dark purple":  "purple",
        "light purple": "lavender",
        "bright yellow":"yellow",
        "bright red":   "red",
        "bright blue":  "blue",
        "bright green": "green",
        "bright orange":"orange",
        "dark orange":  "rust",
    }
    if word in _COMPOUNDS:
        word = _COMPOUNDS[word]

    # Direct hit in our palette.
    if word in _REFERENCE_PALETTE:
        return _REFERENCE_PALETTE[word]

    # Fashion synonym table: maps descriptive terms to the nearest palette entry
    # or to a specific RGB. This table is intentionally opinionated — "burgundy"
    # is closer to maroon than to red perceptually, etc.
    _SYNONYMS: dict[str, Tuple[int, int, int]] = {
        "burgundy":   (128,   0,  32),
        "wine":       (114,   0,  32),
        "crimson":    (180,   0,  32),
        "scarlet":    (200,   0,  30),
        "coral":      (255, 100,  80),
        "salmon":     (250, 130, 110),
        "rust":       (183,  65,  14),
        "tan":        (210, 180, 140),
        "camel":      (195, 150, 100),
        "khaki":      (195, 176, 100),
        "ecru":       (220, 210, 180),
        "ivory":      (255, 255, 220),
        "cream":      (255, 253, 208),
        "off-white":  (240, 235, 220),
        "charcoal":   ( 54,  69,  79),
        "slate":      (112, 128, 144),
        "silver":     (192, 192, 192),
        "cobalt":     ( 0,  71, 171),
        "royal":      ( 65, 105, 225),
        "sky":        (135, 206, 235),
        "denim":      ( 21,  96, 189),
        "indigo":     ( 75,   0, 130),
        "lavender":   (181, 126, 220),
        "violet":     (143,   0, 255),
        "magenta":    (255,   0, 255),
        "fuchsia":    (255,   0, 128),
        "hot pink":   (255, 105, 180),
        "lime":       (180, 230,  50),
        "forest":     ( 34, 139,  34),
        "sage":       (130, 160, 110),
        "mint":       (152, 255, 152),
        "turquoise":  ( 64, 224, 208),
        "gold":       (212, 175,  55),
        "mustard":    (220, 180,  30),
        "champagne":  (247, 231, 206),
        "nude":       (220, 185, 160),
        "mauve":      (224, 176, 255),
        "blush":      (255, 182, 193),
        "taupe":      (144, 128, 112),
    }

    if word in _SYNONYMS:
        return _SYNONYMS[word]

    # CSS4 fallback via webcolors — covers names like "grey", "chocolate", "azure",
    # "plum", "tomato", etc. that aren't in our palette or synonym table.
    # IMPORTANT: name_to_hex() returns a hex *string* ("#ffff00"), not an RGB
    # tuple. Always convert through hex_to_rgb(); never return the hex directly.
    try:
        import webcolors
        hex_val = webcolors.name_to_hex(word)          # raises ValueError if unknown
        rgb = webcolors.hex_to_rgb(hex_val)            # IntegerRGB namedtuple
        return (rgb.red, rgb.green, rgb.blue)
    except Exception:
        return None


def rgb_distance(rgb1: list[int], rgb2: list[int] | Tuple[int, int, int]) -> float:
    """Euclidean distance between two RGB triples."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)))
