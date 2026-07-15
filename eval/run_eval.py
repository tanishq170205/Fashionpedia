"""
Evaluation script for the five benchmark queries.

Runs the full retrieval pipeline on each query, saves a contact sheet (top-5
grid image) for visual inspection, and supports interactive relevance marking
to compute precision@5 and precision@10.

Relevance judgments are persisted to eval/judgments/<slug>.json so that
re-runs don't require re-judging. Delete the judgment files to start fresh.

Usage:
    python run_eval.py --db-path ../chroma_db
    python run_eval.py --db-path ../chroma_db --no-llm   # skip Groq API
    python run_eval.py --db-path ../chroma_db --skip-judgment  # contact sheets only

Precision@K is printed per query and averaged at the end. This gives a real
number to reference in the writeup instead of purely qualitative assessment.

Note: these five queries are eval fixtures. They are hardcoded here because they
are the evaluation protocol, not because the retrieval pipeline knows about them.
The same pipeline will handle any other query identically.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import chromadb

# Resolve paths relative to this file so the script works regardless of CWD.
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_RETRIEVER_DIR = _REPO_ROOT / "retriever"

# The retriever directory must be on sys.path before any retriever imports.
sys.path.insert(0, str(_RETRIEVER_DIR))

from config import RetrieverConfig  # from retriever/config.py
from embedder import load_model as load_clip  # from retriever/embedder.py
from main import run_query  # from retriever/main.py

# contact_sheet lives next to this file (eval/).
sys.path.insert(0, str(_SCRIPT_DIR))
from contact_sheet import make_contact_sheet


# Five benchmark queries. These are the only queries where this file's logic
# is specialized. The retrieval pipeline itself is query-agnostic.
BENCHMARK_QUERIES = [
    "a person in a bright yellow raincoat",
    "professional business attire inside a modern office",
    "someone wearing a blue shirt sitting on a park bench",
    "casual weekend outfit for a city walk",
    "a red tie and a white shirt in a formal setting",
]


def slugify(text: str) -> str:
    """Convert a query string to a safe filename slug."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:60]


def load_or_create_judgment(judgment_path: Path, image_ids: list[str]) -> dict:
    """Load persisted judgments or return an empty dict."""
    if judgment_path.exists():
        with open(judgment_path) as f:
            return json.load(f)
    return {img_id: None for img_id in image_ids}


def interactive_judge(
    query: str,
    results: list,
    judgment_path: Path,
    top_k: int = 10,
) -> dict[str, bool]:
    """
    Prompt the user to mark each returned image as relevant or not.

    Displays image path and score, asks for input. Persists to JSON after each
    answer so partial judgments survive a KeyboardInterrupt.

    Returns a dict mapping image_id → bool (True=relevant).
    """
    image_ids = [r.image_id for r in results[:top_k]]
    judgments = load_or_create_judgment(judgment_path, image_ids)

    # Check if all judgments are already complete.
    unjudged = [img_id for img_id, v in judgments.items() if v is None]
    if not unjudged:
        print(f"  Judgments already complete for this query (loaded from {judgment_path}).")
        return {k: v for k, v in judgments.items() if v is not None}

    print(f"\n  Relevance judgment for: \"{query}\"")
    print(f"  Mark each image [r]elevant, [n]ot relevant, or [s]kip.")
    print()

    for i, result in enumerate(results[:top_k]):
        if judgments.get(result.image_id) is not None:
            continue  # already judged

        img_name = Path(result.image_path).name if result.image_path else result.image_id
        print(
            f"  [{i+1}/{min(top_k, len(results))}] {img_name}  "
            f"score={result.final_score:.3f}  s1={result.stage1_score:.3f}  attr={result.attribute_score:.3f}"
        )
        if result.matched_attributes:
            attrs = ", ".join(
                f"{a['query_label']}({a['query_color']})" for a in result.matched_attributes
            )
            print(f"         matched attributes: {attrs}")

        while True:
            answer = input("  Relevant? [r/n/s]: ").strip().lower()
            if answer in ("r", "relevant", "y", "yes"):
                judgments[result.image_id] = True
                break
            elif answer in ("n", "not", "no"):
                judgments[result.image_id] = False
                break
            elif answer in ("s", "skip", ""):
                judgments[result.image_id] = None
                break
            else:
                print("  Enter r (relevant), n (not relevant), or s (skip).")

        # Persist after each answer so partial runs are not lost.
        judgment_path.parent.mkdir(parents=True, exist_ok=True)
        with open(judgment_path, "w") as f:
            json.dump(judgments, f, indent=2)

    return {k: v for k, v in judgments.items() if v is not None}


def precision_at_k(results: list, judgments: dict[str, bool], k: int) -> float:
    """Compute precision@k given a ranked result list and binary relevance judgments."""
    relevant_count = 0
    judged_count = 0
    for result in results[:k]:
        verdict = judgments.get(result.image_id)
        if verdict is None:
            continue  # skipped — exclude from denominator
        judged_count += 1
        if verdict:
            relevant_count += 1
    if judged_count == 0:
        return float("nan")
    return relevant_count / judged_count


def main() -> None:
    import argparse
    import torch

    parser = argparse.ArgumentParser(description="Evaluate the retrieval pipeline on benchmark queries.")
    parser.add_argument("--db-path", default="../chroma_db", help="Path to ChromaDB directory.")
    parser.add_argument("--clip-model", default="ViT-L/14")
    parser.add_argument("--groq-model", default="llama-3.3-70b-versatile")
    parser.add_argument("--top-k-stage1", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10, dest="top_k_final",
                        help="Number of results to retrieve per query (used for contact sheet and P@K).")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--skip-judgment", action="store_true",
                        help="Generate contact sheets only; do not prompt for relevance judgments.")
    args = parser.parse_args()

    config = RetrieverConfig(
        db_path=args.db_path,
        clip_model=args.clip_model,
        groq_model=args.groq_model,
        top_k_stage1=args.top_k_stage1,
        top_k_final=args.top_k_final,
        no_llm=args.no_llm,
    )

    # Load CLIP once; all queries reuse the same model state.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_clip(config.clip_model, device)

    client = chromadb.PersistentClient(path=args.db_path)
    collection = client.get_collection("fashion_images")
    print(f"Collection has {collection.count()} indexed images.\n")

    results_dir = _SCRIPT_DIR / "results"
    judgments_dir = _SCRIPT_DIR / "judgments"

    all_p5 = []
    all_p10 = []

    for query in BENCHMARK_QUERIES:
        slug = slugify(query)
        print(f"\n{'='*70}")
        print(f"Query: {query!r}")
        print(f"{'='*70}")

        results = run_query(query, config, collection)

        # Contact sheet (top 5 for visual check).
        sheet_path = results_dir / f"{slug}.png"
        make_contact_sheet(
            query=query,
            results=[
                {
                    "image_path": r.image_path,
                    "final_score": r.final_score,
                    "stage1_score": r.stage1_score,
                    "attribute_score": r.attribute_score,
                    "setting_score": r.setting_score,
                }
                for r in results[:5]
            ],
            output_path=str(sheet_path),
        )

        if args.skip_judgment:
            continue

        # Relevance judgment + precision@K.
        judgment_path = judgments_dir / f"{slug}.json"
        judgments = interactive_judge(
            query=query,
            results=results,
            judgment_path=judgment_path,
            top_k=10,
        )

        p5 = precision_at_k(results, judgments, k=5)
        p10 = precision_at_k(results, judgments, k=10)

        print(f"\n  P@5 = {p5:.2f}   P@10 = {p10:.2f}")
        if not (p5 != p5):  # not NaN
            all_p5.append(p5)
        if not (p10 != p10):
            all_p10.append(p10)

    if all_p5 or all_p10:
        print(f"\n{'='*70}")
        print("SUMMARY")
        if all_p5:
            print(f"  Mean P@5  = {sum(all_p5) / len(all_p5):.3f}  ({len(all_p5)} queries judged)")
        if all_p10:
            print(f"  Mean P@10 = {sum(all_p10) / len(all_p10):.3f}  ({len(all_p10)} queries judged)")
        print()


if __name__ == "__main__":
    main()
