"""
Indexer entry point.

Wires together detector, embedder, color extractor, and ChromaDB store.
Run with --help to see all configurable options.

Usage:
    python main.py
    python main.py --image-dir /path/to/images --db-path /path/to/db
    python main.py --max-images 50 --device cpu   # quick sanity check
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from config import parse_args
from color_extractor import extract_dominant_color
from detector import load_detector, detect_garments_and_persons
from embedder import load_model as load_clip, encode_image, encode_images_batch, encode_text
from chroma_store import (
    get_or_create_collection,
    load_indexed_keys,
    upsert_image,
)


def main() -> None:
    config = parse_args()

    # Resolve device early so everything uses the same one.
    import torch
    if config.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU.")
        config.device = "cpu"

    print(f"Image dir : {config.image_dir}")
    print(f"DB path   : {config.db_path}")
    print(f"CLIP model: {config.clip_model} ({config.clip_model_version})")
    print(f"Detector  : {config.detector_model} (threshold={config.detector_threshold})")
    print(f"Device    : {config.device}")

    # --- Load models ---
    load_clip(config.clip_model, config.device)
    detector = load_detector(config.detector_model, config.device)

    # --- Chroma setup ---
    collection = get_or_create_collection(config.db_path)
    already_indexed = load_indexed_keys(collection)
    print(f"Already indexed: {len(already_indexed)} image+model combinations.")

    # --- Discover images ---
    image_dir = Path(config.image_dir)
    all_images = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.jpeg"))
    if config.max_images is not None:
        all_images = all_images[: config.max_images]
    print(f"Total images found: {len(all_images)}")

    # Build the idempotency skip key tuple for this run's config.
    def _skip_key(image_id: str) -> tuple:
        return (image_id, config.clip_model, config.clip_model_version, config.detector_model)

    to_process = [
        img for img in all_images
        if _skip_key(img.stem) not in already_indexed
    ]
    print(f"Images to process : {len(to_process)} (skipping {len(all_images) - len(to_process)} already indexed)")

    if not to_process:
        print("Nothing to do. Exiting.")
        return

    # --- Per-image processing ---
    start_time = time.time()
    stats = {"processed": 0, "skipped_idempotent": len(all_images) - len(to_process), "failed": 0}

    for image_path in tqdm(to_process, desc="Indexing", unit="img"):
        image_id = image_path.stem  # Fashionpedia filenames are content hashes — globally unique

        # Load image — handle corrupt files without crashing the whole run.
        try:
            image = Image.open(image_path).convert("RGB")
        except UnidentifiedImageError:
            print(f"\nERROR: Cannot open {image_path.name} (unrecognized format). Skipping.")
            stats["failed"] += 1
            continue
        except Exception as e:
            print(f"\nERROR: Unexpected error opening {image_path.name}: {e}. Skipping.")
            stats["failed"] += 1
            continue

        # Resize large images before detection — GDINO's attention cost scales
        # quadratically with image tokens, so capping at 800px on the long side
        # meaningfully reduces CPU time without hurting detection quality.
        MAX_SIDE = 800
        if max(image.size) > MAX_SIDE:
            scale = MAX_SIDE / max(image.size)
            new_size = (int(image.width * scale), int(image.height * scale))
            image = image.resize(new_size, Image.LANCZOS)

        # --- Detection ---
        detection_result = detect_garments_and_persons(
            detector, image, threshold=config.detector_threshold
        )
        raw_garments = detection_result["garments"]

        # --- Build crop list and batch everything through CLIP in one pass ---
        img_w, img_h = image.size
        valid_crops = []      # (garment_dict, PIL_crop) pairs that passed bbox validation

        for garment in raw_garments:
            x1, y1, x2, y2 = garment["bbox"]
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(img_w, x2); y2 = min(img_h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image.crop((x1, y1, x2, y2))
            garment["bbox"] = [x1, y1, x2, y2]  # update with clamped coords
            valid_crops.append((garment, crop))

        # One CLIP forward pass for full image + all crops combined.
        # This is the key CPU speedup: N+1 calls → 1 call.
        all_pil = [image] + [crop for _, crop in valid_crops]
        all_embeddings = encode_images_batch(all_pil)
        full_embedding = all_embeddings[0]
        crop_embeddings = all_embeddings[1:]

        # --- Per-region: color extraction (still per-crop, but cheap) ---
        regions = []
        for (garment, crop), region_embedding in zip(valid_crops, crop_embeddings):
            try:
                color_info = extract_dominant_color(crop, n_clusters=config.color_clusters)
            except ValueError:
                color_info = {"color_rgb": [128, 128, 128], "color_name": "unknown"}

            regions.append({
                "label":           garment["label"],
                "bbox":            garment["bbox"],
                "score":           garment["score"],
                "person_id":       garment["person_id"],
                "color_rgb":       color_info["color_rgb"],
                "color_name":      color_info["color_name"],
                "region_embedding": region_embedding,
            })

        # --- Store in ChromaDB ---
        upsert_image(
            collection=collection,
            image_id=image_id,
            image_path=str(image_path.resolve()),
            full_image_embedding=full_embedding,
            regions=regions,
            clip_model=config.clip_model,
            clip_model_version=config.clip_model_version,
            detector_model=config.detector_model,
        )

        stats["processed"] += 1

    # --- Summary ---
    elapsed = time.time() - start_time
    summary = {
        "processed":         stats["processed"],
        "skipped_idempotent": stats["skipped_idempotent"],
        "failed":            stats["failed"],
        "elapsed_seconds":   round(elapsed, 1),
        "config": {
            "image_dir":          config.image_dir,
            "clip_model":         config.clip_model,
            "clip_model_version": config.clip_model_version,
            "detector_model":     config.detector_model,
            "detector_threshold": config.detector_threshold,
        },
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }

    summary_path = Path(config.db_path) / "index_run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Processed={stats['processed']}, Skipped={stats['skipped_idempotent']}, Failed={stats['failed']}")
    print(f"Elapsed: {elapsed:.0f}s. Summary written to {summary_path}")


if __name__ == "__main__":
    main()
