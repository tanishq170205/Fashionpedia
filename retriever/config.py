"""
Retriever configuration.

All weights and thresholds are exposed here. The score fusion weights must
sum to 1.0 — this is validated at startup in main.py.

Default weight rationale:
  w_stage1=0.50: The full-image CLIP embedding captures overall scene and
    garment gestalt well. It should dominate for queries without explicit
    attribute structure ("casual weekend outfit").
  w_attribute=0.35: Attribute matching is the main reason we built this
    pipeline instead of just using plain CLIP. It needs enough weight to
    reorder candidates when the stage-1 similarity is close.
  w_setting=0.15: Setting/context matching is noisy because we're comparing
    a short phrase embedding against a full-scene embedding; it helps at the
    margin but shouldn't dominate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class RetrieverConfig:
    # --- Storage ---
    db_path: str = "./chroma_db"

    # --- Models ---
    clip_model: str = "ViT-L/14"
    groq_model: str = "llama-3.3-70b-versatile"

    # --- Retrieval parameters ---
    # How many candidates to pull in stage 1. 100 is a reasonable default:
    # large enough that most true positives are in the candidate set,
    # small enough that stage-2 reranking finishes in <1s on CPU.
    top_k_stage1: int = 100
    top_k_final: int = 5

    # --- Score fusion weights (must sum to 1.0) ---
    w_stage1: float = 0.50
    w_attribute: float = 0.35
    w_setting: float = 0.15

    # --- Matching thresholds ---
    # Euclidean RGB distance below which two colors are considered a match.
    # 60 is approximately 14% of the 0-441 range for RGB space.
    # It's deliberately generous because color naming and measurement both
    # introduce ~10-15% of error independently.
    color_distance_threshold: float = 60.0

    # CLIP cosine similarity threshold for garment label matching.
    # Cross-modal CLIP cosines (text vs image) for true matches typically land
    # in the 0.20-0.35 range. 0.25 is chosen to accept genuine matches while
    # rejecting completely unrelated regions.
    garment_similarity_threshold: float = 0.25

    # --- LLM fallback ---
    # If True, skip the Groq API call and use simple keyword extraction instead.
    # Useful when GROQ_API_KEY is not set or for offline testing.
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
    )

    total_weight = cfg.w_stage1 + cfg.w_attribute + cfg.w_setting
    if abs(total_weight - 1.0) > 1e-6:
        raise ValueError(
            f"Score fusion weights must sum to 1.0, got {total_weight:.4f}. "
            f"Adjust --w-stage1, --w-attribute, --w-setting."
        )

    return cfg
