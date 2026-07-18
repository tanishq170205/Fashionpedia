"""
eval/calibrate_thresholds.py — Task 3

Empirically calibrates two thresholds in retriever/config.py by sampling real
region data from a ChromaDB index and measuring the CLIP similarity / RGB color
distance distributions for positive (true match) and negative (wrong match) pairs.

Outputs:
  - Garment threshold: midpoint between P10 of positive cosines and P90 of negative cosines.
  - Color threshold  : midpoint between P90 of positive distances and P10 of negative distances.

Usage:
    cd Fashionpedia
    python eval/calibrate_thresholds.py --db-path ./chroma_db --clip-model ViT-B/32
    python eval/calibrate_thresholds.py --db-path ./chroma_db_fashion --clip-model hf-hub:Marqo/marqo-fashionCLIP
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import chromadb

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT   = _SCRIPT_DIR.parent

sys.path.insert(0, str(_REPO_ROOT / "retriever"))
sys.path.insert(0, str(_REPO_ROOT / "indexer"))

from embedder import load_model as load_clip, encode_text   # retriever/embedder.py
from chroma_store import load_regions                        # indexer/chroma_store.py
from color_extractor import color_name_to_rgb, rgb_distance # indexer/color_extractor.py


# ── Common fashion garment labels — used to generate hard negatives ──────────
GARMENT_LABELS = [
    "shirt", "dress", "jacket", "coat", "trousers", "skirt", "shorts",
    "blouse", "suit", "sweater", "hoodie", "blazer", "jeans", "vest",
    "cardigan", "leggings", "jumpsuit", "tuxedo", "raincoat", "overcoat",
    "top", "t-shirt", "tie", "scarf", "hat", "bag", "shoes", "boots",
]

COLOR_NAMES = [
    "red", "blue", "green", "yellow", "black", "white", "gray", "brown",
    "orange", "pink", "purple", "navy", "beige", "cream", "gold", "silver",
]


def sample_regions_from_db(
    db_path: str,
    n_images: int = 100,
    seed: int = 42,
) -> list[dict]:
    """
    Sample up to n_images documents from the ChromaDB collection and return a
    flat list of region dicts (each region has: label, color_name, color_rgb,
    region_embedding).
    """
    client     = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection("fashion_images")
    total      = collection.count()
    print(f"  Collection: {total} images indexed at {db_path}")

    # Fetch all document IDs, then sample.
    all_ids = collection.get(include=[])["ids"]
    rng     = random.Random(seed)
    sampled = rng.sample(all_ids, min(n_images, len(all_ids)))

    result = collection.get(ids=sampled, include=["metadatas"])

    regions = []
    for meta in result["metadatas"]:
        for r in load_regions(meta):
            if r.get("region_embedding") is not None and r.get("label"):
                regions.append(r)

    print(f"  Sampled {len(regions)} regions from {len(sampled)} images.")
    return regions


def calibrate_garment_threshold(
    regions: list[dict],
    n_pairs: int = 40,
    seed: int = 42,
) -> dict:
    """
    Build positive pairs (region_embedding vs CLIP(correct_label)) and
    negative pairs (region_embedding vs CLIP(wrong_label)).

    Returns a dict with distributions and recommended threshold.
    """
    rng = random.Random(seed)

    # Pre-compute text embeddings for all unique labels in the sample.
    unique_labels = list({r["label"] for r in regions if r.get("label")})
    print(f"\n  Encoding {len(unique_labels)} unique garment labels...")
    label_vecs: dict[str, np.ndarray] = {}
    for lbl in unique_labels:
        label_vecs[lbl] = encode_text(lbl)

    # Also encode some additional garment labels for hard negatives.
    extra_labels = [l for l in GARMENT_LABELS if l not in label_vecs]
    for lbl in extra_labels:
        label_vecs[lbl] = encode_text(lbl)

    all_labels = list(label_vecs.keys())

    # Sample positive pairs.
    pos_regions = rng.sample(regions, min(n_pairs, len(regions)))
    positive_sims = []
    for r in pos_regions:
        lbl = r["label"]
        if lbl not in label_vecs:
            continue
        emb = r["region_embedding"]
        rv  = emb / (np.linalg.norm(emb) + 1e-10)
        lv  = label_vecs[lbl] / (np.linalg.norm(label_vecs[lbl]) + 1e-10)
        sim = float(np.dot(lv, rv))
        positive_sims.append(sim)

    # Sample negative pairs (wrong label for each region).
    neg_regions = rng.sample(regions, min(n_pairs, len(regions)))
    negative_sims = []
    for r in neg_regions:
        lbl = r["label"]
        wrong = rng.choice([l for l in all_labels if l != lbl])
        emb   = r["region_embedding"]
        rv    = emb / (np.linalg.norm(emb) + 1e-10)
        lv    = label_vecs[wrong] / (np.linalg.norm(label_vecs[wrong]) + 1e-10)
        sim   = float(np.dot(lv, rv))
        negative_sims.append(sim)

    pos_arr = np.array(positive_sims)
    neg_arr = np.array(negative_sims)

    # Calibrated threshold = midpoint between P10 of positives and P90 of negatives.
    p10_pos = float(np.percentile(pos_arr, 10)) if len(pos_arr) > 0 else 0.20
    p90_neg = float(np.percentile(neg_arr, 90)) if len(neg_arr) > 0 else 0.20
    threshold = (p10_pos + p90_neg) / 2.0

    return {
        "positive_sims": positive_sims,
        "negative_sims": negative_sims,
        "pos_mean": float(np.mean(pos_arr)) if len(pos_arr) else None,
        "pos_p10":  p10_pos,
        "pos_p50":  float(np.percentile(pos_arr, 50)) if len(pos_arr) else None,
        "neg_mean": float(np.mean(neg_arr)) if len(neg_arr) else None,
        "neg_p90":  p90_neg,
        "neg_p50":  float(np.percentile(neg_arr, 50)) if len(neg_arr) else None,
        "recommended_threshold": round(threshold, 3),
    }


def calibrate_color_threshold(
    regions: list[dict],
    n_pairs: int = 40,
    seed: int = 42,
) -> dict:
    """
    Build positive pairs (stored_rgb vs color_name_to_rgb(stored color_name)) and
    negative pairs (stored_rgb vs color_name_to_rgb(wrong_color)).

    Returns a dict with distributions and recommended threshold.
    """
    rng = random.Random(seed)

    # Filter to regions that have usable color data.
    colored = [
        r for r in regions
        if r.get("color_name") and r.get("color_rgb") and len(r["color_rgb"]) == 3
    ]
    print(f"\n  Regions with color data: {len(colored)}")

    pos_dists = []
    for r in rng.sample(colored, min(n_pairs, len(colored))):
        ref = color_name_to_rgb(r["color_name"])
        if ref is None:
            continue
        dist = rgb_distance(r["color_rgb"], ref)
        pos_dists.append(dist)

    neg_dists = []
    for r in rng.sample(colored, min(n_pairs, len(colored))):
        wrong_color = rng.choice([c for c in COLOR_NAMES if c != r.get("color_name")])
        ref = color_name_to_rgb(wrong_color)
        if ref is None:
            continue
        dist = rgb_distance(r["color_rgb"], ref)
        neg_dists.append(dist)

    pos_arr = np.array(pos_dists)
    neg_arr = np.array(neg_dists)

    # Calibrated threshold = midpoint between P90 of positives and P10 of negatives.
    p90_pos = float(np.percentile(pos_arr, 90)) if len(pos_arr) > 0 else 80.0
    p10_neg = float(np.percentile(neg_arr, 10)) if len(neg_arr) > 0 else 80.0
    threshold = (p90_pos + p10_neg) / 2.0

    return {
        "positive_dists": pos_dists,
        "negative_dists": neg_dists,
        "pos_mean": float(np.mean(pos_arr)) if len(pos_arr) else None,
        "pos_p50":  float(np.percentile(pos_arr, 50)) if len(pos_arr) else None,
        "pos_p90":  p90_pos,
        "neg_mean": float(np.mean(neg_arr)) if len(neg_arr) else None,
        "neg_p50":  float(np.percentile(neg_arr, 50)) if len(neg_arr) else None,
        "neg_p10":  p10_neg,
        "recommended_threshold": round(threshold, 3),
    }


def print_summary(garment_res: dict, color_res: dict) -> None:
    print("\n" + "="*60)
    print("GARMENT SIMILARITY CALIBRATION")
    print("="*60)
    print(f"  Positive pairs (correct label vs region_embedding):")
    print(f"    mean={garment_res['pos_mean']:.3f}  P10={garment_res['pos_p10']:.3f}  P50={garment_res['pos_p50']:.3f}")
    print(f"  Negative pairs (wrong label vs region_embedding):")
    print(f"    mean={garment_res['neg_mean']:.3f}  P50={garment_res['neg_p50']:.3f}  P90={garment_res['neg_p90']:.3f}")
    print(f"  -> Recommended garment_similarity_threshold = {garment_res['recommended_threshold']:.3f}")
    print(f"    (midpoint between pos P10={garment_res['pos_p10']:.3f} and neg P90={garment_res['neg_p90']:.3f})")

    print("\n" + "="*60)
    print("COLOR DISTANCE CALIBRATION")
    print("="*60)
    print(f"  Positive pairs (correct color vs stored RGB):")
    print(f"    mean={color_res['pos_mean']:.1f}  P50={color_res['pos_p50']:.1f}  P90={color_res['pos_p90']:.1f}")
    print(f"  Negative pairs (wrong color vs stored RGB):")
    print(f"    mean={color_res['neg_mean']:.1f}  P10={color_res['neg_p10']:.1f}  P50={color_res['neg_p50']:.1f}")
    print(f"  -> Recommended color_distance_threshold = {color_res['recommended_threshold']:.1f}")
    print(f"    (midpoint between pos P90={color_res['pos_p90']:.1f} and neg P10={color_res['neg_p10']:.1f})")

    print("\n" + "="*60)
    print("ACTION REQUIRED")
    print("="*60)
    print("  Update retriever/config.py with the values above:")
    print(f"    garment_similarity_threshold: float = {garment_res['recommended_threshold']:.3f}")
    print(f"    color_distance_threshold: float = {color_res['recommended_threshold']:.1f}")


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description="Calibrate retrieval thresholds from real indexed data.")
    parser.add_argument("--db-path",    default="../chroma_db",  help="Path to ChromaDB directory.")
    parser.add_argument("--clip-model", default="ViT-B/32",      help="CLIP model name (must match index).")
    parser.add_argument("--n-images",   type=int, default=150,   help="Images to sample for calibration.")
    parser.add_argument("--n-pairs",    type=int, default=50,    help="Positive/negative pairs per threshold.")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--out",        default=None,            help="Optional JSON output path.")
    args = parser.parse_args()

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"Loading CLIP model '{args.clip_model}' on {device}...")
    load_clip(args.clip_model, device)

    print(f"\nSampling regions from {args.db_path}...")
    regions = sample_regions_from_db(args.db_path, n_images=args.n_images, seed=args.seed)

    if len(regions) < 10:
        print("ERROR: fewer than 10 regions found — is the db_path correct and non-empty?")
        sys.exit(1)

    print("\nRunning garment similarity calibration...")
    garment_res = calibrate_garment_threshold(regions, n_pairs=args.n_pairs, seed=args.seed)

    print("\nRunning color distance calibration...")
    color_res = calibrate_color_threshold(regions, n_pairs=args.n_pairs, seed=args.seed)

    print_summary(garment_res, color_res)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"garment": garment_res, "color": color_res}, f, indent=2)
        print(f"\n  Full results written to {out_path}")


if __name__ == "__main__":
    main()
