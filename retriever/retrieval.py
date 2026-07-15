"""
Stage 1 retrieval: ANN search using ChromaDB's HNSW index.

This module only does the fast ANN pass. It does not touch region metadata,
does not compute attribute matches, and does not look at the full collection.
That discipline is what makes the pipeline scalable: if the corpus grows from
3,200 to 1,000,000 images, stage 1 is still a single HNSW lookup and stage 2
still only touches the top-k candidates.

Why top_k=100 by default:
  - At 100 candidates, stage-2 reranking (decoding base64 embeddings, computing
    cosine similarities) takes ~0.2-0.5s on CPU for ViT-L/14 embeddings.
  - At 3,200 images, a true positive is almost certainly in the top 100 for any
    reasonable query, so increasing this beyond 100 adds latency without recall
    improvement. At 1M images you might want to raise it to 500.
  - The minimum useful value is roughly 5-10x the final top_k, to give the
    reranker enough headroom to reorder.
"""

from __future__ import annotations

import numpy as np
import chromadb


def stage1_retrieve(
    query_embedding: np.ndarray,
    collection: chromadb.Collection,
    top_k: int = 100,
) -> list[dict]:
    """
    Query the HNSW index and return the top_k candidates.

    Returns a list of dicts, one per candidate:
        {
          "id":       str,
          "distance": float,   # raw Chroma distance (cosine space: 0=identical, 2=opposite)
          "metadata": dict,
          "embedding": list[float],  # stored full-image embedding
        }

    The raw distance is passed through unchanged. Converting to similarity
    (1 - distance) happens in the reranker, not here, so the raw value is
    always visible for debugging.
    """
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(top_k, collection.count()),
        include=["metadatas", "distances", "embeddings"],
    )

    candidates = []
    ids = results["ids"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]
    embeddings = results["embeddings"][0]

    for doc_id, dist, meta, emb in zip(ids, distances, metadatas, embeddings):
        candidates.append({
            "id":        doc_id,
            "distance":  dist,
            "metadata":  meta,
            "embedding": emb,   # full-image embedding as a Python list
        })

    return candidates
