"""
Indexer configuration.

All paths and model choices are exposed here; nothing is hardcoded in the
modules that consume them. The defaults point at the Fashionpedia val_test2020
layout, but the indexer should work on any folder of JPGs by passing --image-dir.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class IndexerConfig:
    # --- Data paths ---
    image_dir: str = "./datasets/val_test2020/test"
    db_path: str = "./chroma_db"

    # --- Model choices ---
    # CLIP model name. Two backends are supported (see indexer/embedder.py):
    #   OpenAI CLIP  : "ViT-B/32", "ViT-L/14", "ViT-B/16", "ViT-L/14@336px"
    #   open_clip    : any "hf-hub:org/model" identifier
    #
    # Fashion-domain checkpoints (strongly recommended for better retrieval):
    #   "hf-hub:Marqo/marqo-fashionCLIP"       ← best-in-class for garment text-image alignment
    #   "hf-hub:patrickjohncyh/fashion-clip"   ← alternative, slightly smaller
    #
    # NOTE: changing this requires re-indexing. The mismatch guard in app/main.py
    # will refuse to start if the stored clip_model in the collection doesn't match.
    clip_model: str = "ViT-B/32"
    # Version tag stored in metadata; bump if you swap to a fine-tuned checkpoint.
    clip_model_version: str = "openai"

    # Detector backend: "groundingdino" or "owlvit".
    # "groundingdino" is tried first; owlvit is the explicit fallback.
    detector_model: str = "groundingdino"

    # --- Detector settings ---
    # Minimum confidence to keep a detection box.
    # 0.30 is conservative but intentional: GDINO boxes below this threshold on
    # fashion imagery are almost always background texture or partial occlusions
    # that would add noise to the reranker without contributing real signal.
    detector_threshold: float = 0.30

    # --- Processing ---
    batch_size: int = 16
    device: str = "cuda"   # falls back to "cpu" at runtime if CUDA unavailable

    # K-means clusters for dominant color extraction.
    color_clusters: int = 5

    # Cap for debugging / quick sanity checks. None = process all images.
    max_images: Optional[int] = None


def parse_args() -> IndexerConfig:
    parser = argparse.ArgumentParser(
        description="Index Fashionpedia images into ChromaDB with garment detection and CLIP embeddings."
    )
    parser.add_argument(
        "--image-dir",
        default=IndexerConfig.image_dir,
        help="Path to folder containing .jpg images.",
    )
    parser.add_argument(
        "--db-path",
        default=IndexerConfig.db_path,
        help="Path where ChromaDB will persist its data.",
    )
    parser.add_argument(
        "--clip-model",
        default=IndexerConfig.clip_model,
        help=(
            "CLIP backbone to use for image/region embeddings. "
            "OpenAI CLIP names: ViT-B/32, ViT-B/16, ViT-L/14. "
            "Fashion-domain (requires open_clip_torch): "
            "hf-hub:Marqo/marqo-fashionCLIP, hf-hub:patrickjohncyh/fashion-clip. "
            "Changing this requires re-indexing the full dataset."
        ),
    )
    parser.add_argument(
        "--clip-model-version",
        default=IndexerConfig.clip_model_version,
        help="Version tag for the CLIP checkpoint (used for idempotency checks).",
    )
    parser.add_argument(
        "--detector-model",
        default=IndexerConfig.detector_model,
        choices=["groundingdino", "owlvit"],
        help="Object detector backend.",
    )
    parser.add_argument(
        "--detector-threshold",
        type=float,
        default=IndexerConfig.detector_threshold,
        help="Minimum detection confidence score (0-1).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=IndexerConfig.batch_size,
        help="Number of images to process before flushing to ChromaDB.",
    )
    parser.add_argument(
        "--device",
        default=IndexerConfig.device,
        choices=["cuda", "cpu"],
        help="Torch device for model inference.",
    )
    parser.add_argument(
        "--color-clusters",
        type=int,
        default=IndexerConfig.color_clusters,
        help="Number of K-means clusters for dominant color extraction.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Process at most N images (useful for quick tests). Omit for full run.",
    )

    args = parser.parse_args()
    return IndexerConfig(
        image_dir=args.image_dir,
        db_path=args.db_path,
        clip_model=args.clip_model,
        clip_model_version=args.clip_model_version,
        detector_model=args.detector_model,
        detector_threshold=args.detector_threshold,
        batch_size=args.batch_size,
        device=args.device,
        color_clusters=args.color_clusters,
        max_images=args.max_images,
    )
