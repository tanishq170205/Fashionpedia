"""
Retriever configuration — single source of truth for all weights and thresholds.

Default weight rationale (after _resolve_weights redistribution):
  w_stage1=0.35: Global CLIP gives direction; attribute matching refines.
  w_attribute=0.50: The core innovation — per-person garment+color matching.
  w_setting=0.15: Noisy scene signal, useful at the margin only.

Thresholds (garment_similarity_threshold, color_distance_threshold) are marked
"PENDING CALIBRATION" — run eval/calibrate_thresholds.py after each re-index
to replace these with measured values from the actual embedding distribution.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class RetrieverConfig:
    # --- Storage ---
    db_path: str = "./chroma_db_fashion"

    # --- Models ---
    # Fashion-domain CLIP checkpoint (hf-hub format → loaded via open_clip_torch).
    # Outperforms vanilla OpenAI ViT-B/32 on garment text-image alignment;
    # directly addresses the assignment's "better than vanilla CLIP" bar.
    # IMPORTANT: changing this requires re-indexing (embeddings are not comparable
    # across models). The mismatch guard in app/main.py enforces this at startup.
    clip_model: str = "hf-hub:Marqo/marqo-fashionCLIP"
    clip_model_version: str = "marqo-fashionclip-v1"
    groq_model: str = "llama-3.3-70b-versatile"

    # --- Retrieval parameters ---
    top_k_stage1: int = 200   # raised from 100: more headroom for rare-attribute queries
    top_k_final: int = 5

    # --- Score fusion weights (must sum to 1.0) ---
    w_stage1: float = 0.35
    w_attribute: float = 0.50
    w_setting: float = 0.15

    # --- Matching thresholds (calibrated 2026-07-17 on ViT-B/32 / chroma_db) ---
    # Source: eval/calibrate_thresholds.py -- 150 images, 50 pairs each.
    # Re-run after any re-index: python eval/calibrate_thresholds.py --db-path <new_db> --clip-model <model>
    #
    # Garment similarity (CLIP cosine, image-crop vs text label):
    #   Positive pairs: mean=0.237  P10=0.212  P50=0.242
    #   Negative pairs: mean=0.215  P50=0.214  P90=0.233
    #   Threshold = midpoint(pos_P10=0.212, neg_P90=0.233) = 0.223
    #   Note: distributions overlap significantly -- vanilla ViT-B/32 has weak
    #   fashion garment discrimination. FashionCLIP is expected to separate them
    #   more cleanly; re-calibrate after re-index.
    garment_similarity_threshold: float = 0.223

    # Color distance (Euclidean RGB, stored region color vs query color ref):
    #   Positive pairs: mean=28.2  P50=22.4  P90=57.3
    #   Negative pairs: mean=211.0  P10=113.3  P50=203.2
    #   Threshold = midpoint(pos_P90=57.3, neg_P10=113.3) = 85.3
    #   Clear separation -- color is a reliable signal. Value raised slightly
    #   from previous estimate of 80.0.
    color_distance_threshold: float = 85.3

    # --- Gap scoring (Task 4) ---
    # When True, garment match contributions are scaled by a confidence factor
    # derived from the gap between the best and second-best region similarity.
    # A decisive match (large gap) counts fully; a marginal one counts partially.
    # Disabling reverts to the original binary pass/fail behaviour for A/B comparison.
    use_gap_scoring: bool = True

    # --- LLM fallback ---
    no_llm: bool = False


def parse_args() -> RetrieverConfig:
    parser = argparse.ArgumentParser(
        description="Query the fashion image index with a natural language search string."
    )
    parser.add_argument("--query", type=str, required=True, help="Search query string.")
    parser.add_argument("--db-path", default=RetrieverConfig.db_path)
    parser.add_argument("--clip-model", default=RetrieverConfig.clip_model)
    parser.add_argument("--groq-model", default=RetrieverConfig.groq_model)
    parser.add_argument("--top-k-stage1", type=int, default=RetrieverConfig.top_k_stage1)
    parser.add_argument("--top-k", type=int, default=RetrieverConfig.top_k_final, dest="top_k_final")
    parser.add_argument("--w-stage1", type=float, default=RetrieverConfig.w_stage1)
    parser.add_argument("--w-attribute", type=float, default=RetrieverConfig.w_attribute)
    parser.add_argument("--w-setting", type=float, default=RetrieverConfig.w_setting)
    parser.add_argument(
        "--color-threshold",
        type=float,
        default=RetrieverConfig.color_distance_threshold,
        dest="color_distance_threshold",
    )
    parser.add_argument(
        "--garment-threshold",
        type=float,
        default=RetrieverConfig.garment_similarity_threshold,
        dest="garment_similarity_threshold",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        default=False,
        help="Skip Groq LLM parsing and fall back to keyword extraction.",
    )
    parser.add_argument(
        "--no-gap-scoring",
        action="store_true",
        default=False,
        help="Disable gap-scoring; revert to binary pass/fail garment matching for A/B comparison.",
    )

    args = parser.parse_args()

    cfg = RetrieverConfig(
        db_path=args.db_path,
        clip_model=args.clip_model,
        groq_model=args.groq_model,
        top_k_stage1=args.top_k_stage1,
        top_k_final=args.top_k_final,
        w_stage1=args.w_stage1,
        w_attribute=args.w_attribute,
        w_setting=args.w_setting,
        color_distance_threshold=args.color_distance_threshold,
        garment_similarity_threshold=args.garment_similarity_threshold,
        no_llm=args.no_llm,
        use_gap_scoring=not args.no_gap_scoring,
    )

    total_weight = cfg.w_stage1 + cfg.w_attribute + cfg.w_setting
    if abs(total_weight - 1.0) > 1e-6:
        raise ValueError(
            f"Score fusion weights must sum to 1.0, got {total_weight:.4f}. "
            f"Adjust --w-stage1, --w-attribute, --w-setting."
        )

    return cfg
