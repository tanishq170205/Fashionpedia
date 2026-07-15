"""
Retriever entry point.

Wires query parsing, stage-1 ANN retrieval, and stage-2 reranking into a
single query pipeline. Prints ranked results with a per-image score breakdown.

Usage:
    python main.py --query "a person in a bright yellow raincoat"
    python main.py --query "red tie and white shirt" --top-k 10 --no-llm
"""

from __future__ import annotations

import os
import sys

import chromadb

sys.path.insert(0, os.path.dirname(__file__))

from config import parse_args, RetrieverConfig
from embedder import load_model as load_clip, encode_text
from query_parser import parse_query
from retrieval import stage1_retrieve
from reranker import rerank


def run_query(query: str, config, collection: chromadb.Collection) -> list:
    """
    Execute the full two-stage retrieval pipeline for a single query string.
    Returns a list of RankedResult objects (top config.top_k_final).

    This function is importable by eval/run_eval.py so it can run multiple
    queries without reloading the model each time.
    """
    # Parse query into structured fields.
    if config.no_llm:
        from query_parser import _keyword_fallback, ParsedQuery
        parsed = _keyword_fallback(query)
    else:
        parsed = parse_query(query, groq_model=config.groq_model)

    if not parsed.llm_succeeded:
        print(f"  [query_parser] LLM unavailable; using keyword fallback.")

    print(f"  Parsed: garments={parsed.garments}, setting={parsed.setting!r}")

    # Stage 1: CLIP text embedding → HNSW lookup.
    query_embedding = encode_text(query)
    candidates = stage1_retrieve(query_embedding, collection, top_k=config.top_k_stage1)
    print(f"  Stage 1: {len(candidates)} candidates retrieved.")

    # Stage 2: attribute reranking.
    ranked = rerank(
        candidates=candidates,
        parsed_query=parsed,
        query_embedding=query_embedding,
        w_stage1=config.w_stage1,
        w_attribute=config.w_attribute,
        w_setting=config.w_setting,
        color_distance_threshold=config.color_distance_threshold,
        garment_similarity_threshold=config.garment_similarity_threshold,
    )

    return ranked[: config.top_k_final]


def main() -> None:
    import argparse
    import torch

    # Parse the query string separately from the config dataclass, since the
    # query is a per-invocation argument and not a persistent configuration.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--query", type=str, required=True)
    pre_args, _ = pre_parser.parse_known_args()
    query = pre_args.query

    config = parse_args()

    # Load CLIP — CPU is fine here; we only encode text and do dot products.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_clip(config.clip_model, device)

    client = chromadb.PersistentClient(path=config.db_path)
    collection = client.get_collection("fashion_images")

    print(f"\nQuery: {query!r}")
    print("-" * 60)

    results = run_query(query, config, collection)

    print(f"\nTop {len(results)} results:")
    print(f"{'Rank':<5} {'Score':>6} {'S1':>6} {'Attr':>6} {'Set':>6}  {'Image'}")
    print("-" * 80)
    for rank, r in enumerate(results, 1):
        img_name = os.path.basename(r.image_path)
        print(
            f"{rank:<5} {r.final_score:>6.3f} {r.stage1_score:>6.3f} "
            f"{r.attribute_score:>6.3f} {r.setting_score:>6.3f}  {img_name}"
        )
        if r.matched_attributes:
            for attr in r.matched_attributes:
                print(
                    f"       matched: {attr['query_label']}({attr['query_color']}) "
                    f"→ {attr['matched_region_label']}({attr['matched_region_color']}) "
                    f"garment_sim={attr['garment_similarity']}, "
                    f"color_dist={attr['color_distance']}, "
                    f"person_id={attr['person_id']}"
                )


if __name__ == "__main__":
    main()
