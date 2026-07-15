"""
ChromaDB storage layer for the indexer.

Design decisions documented here because they affect retrieval correctness:

1. Cosine distance space is set explicitly at collection creation time.
   ChromaDB defaults to L2 if you don't specify. With L2 distance, the raw
   distance value is not bounded and mixing it with a 0-1 attribute score
   in a weighted sum silently inverts part of the ranking. With cosine space,
   distance ∈ [0, 2] for normalized vectors (0 = identical, 2 = opposite),
   so 1 - distance gives a clean similarity in [-1, 1], clipped to [0, 1].

2. Region embeddings are stored as base64-encoded float32 bytes inside the
   metadata dict, not as separate Chroma documents. ChromaDB metadata values
   must be scalars (str/int/float). A separate collection per region would
   work but complicates the retrieval join — you'd need to look up regions by
   image_id for each of the 100 stage-1 candidates. Inline storage keeps all
   region data co-located with the image document for free.

3. Idempotency is model-version-aware. Skipping by filename alone allows
   silent mixed-model collections if the CLIP checkpoint is changed between
   runs. We store clip_model, clip_model_version, and detector_model in
   metadata and check all three as part of the skip key. If any field
   mismatches, the document is re-indexed and the old embedding is overwritten.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from typing import Optional

import chromadb
import numpy as np


COLLECTION_NAME = "fashion_images"


def get_or_create_collection(db_path: str) -> chromadb.Collection:
    """
    Connect to (or create) the ChromaDB persistent store and return the
    fashion_images collection.

    The collection is created with cosine distance. If it already exists with
    a different distance setting, Chroma will use the existing setting without
    error — this is a known ChromaDB limitation. A warning is printed but
    indexing continues; fixing the distance metric requires deleting the
    collection and re-indexing.
    """
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def load_indexed_keys(collection: chromadb.Collection) -> set[tuple]:
    """
    Return the set of (image_id, clip_model, clip_model_version, detector_model)
    tuples already present in the collection.

    Fetching all IDs is cheap — Chroma stores them in memory. Fetching all
    metadata is slower at large scale (>100k docs), but at 3200 images this
    is negligible and keeps the logic simple.
    """
    try:
        result = collection.get(include=["metadatas"])
    except Exception:
        return set()

    keys = set()
    for doc_id, meta in zip(result["ids"], result["metadatas"]):
        keys.add((
            doc_id,
            meta.get("clip_model", ""),
            meta.get("clip_model_version", ""),
            meta.get("detector_model", ""),
        ))
    return keys


def upsert_image(
    collection: chromadb.Collection,
    image_id: str,
    image_path: str,
    full_image_embedding: np.ndarray,
    regions: list[dict],
    clip_model: str,
    clip_model_version: str,
    detector_model: str,
) -> None:
    """
    Upsert one image document into the collection.

    Each region dict is expected to have:
        label, color_name, color_rgb, bbox, score, person_id, region_embedding

    region_embedding (np.ndarray) is base64-encoded here before storage.
    """
    # Serialize regions: encode the np.ndarray embedding as base64 bytes.
    serialized_regions = []
    for r in regions:
        region_dict = {
            "label":       r["label"],
            "color_name":  r["color_name"],
            "color_rgb":   r["color_rgb"],       # [R, G, B] ints
            "bbox":        r["bbox"],              # [x1, y1, x2, y2]
            "score":       r["score"],
            "person_id":   r["person_id"],
            "region_embedding_b64": _encode_embedding(r["region_embedding"]),
        }
        serialized_regions.append(region_dict)

    metadata = {
        "image_path":         image_path,
        "clip_model":         clip_model,
        "clip_model_version": clip_model_version,
        "detector_model":     detector_model,
        "regions":            json.dumps(serialized_regions),
        "region_count":       len(serialized_regions),
        "indexed_at":         datetime.now(timezone.utc).isoformat(),
    }

    collection.upsert(
        ids=[image_id],
        embeddings=[full_image_embedding.tolist()],
        metadatas=[metadata],
    )


def load_regions(metadata: dict) -> list[dict]:
    """
    Deserialize the regions field from a Chroma metadata dict.

    Returns a list of region dicts with region_embedding decoded back to
    np.ndarray. Returns an empty list if the field is missing or malformed.
    """
    raw = metadata.get("regions", "[]")
    try:
        regions = json.loads(raw)
    except json.JSONDecodeError:
        return []

    for r in regions:
        b64 = r.pop("region_embedding_b64", None)
        if b64 is not None:
            r["region_embedding"] = _decode_embedding(b64)
        else:
            r["region_embedding"] = None

    return regions


def _encode_embedding(vec: np.ndarray) -> str:
    """Serialize a float32 numpy vector to a base64 string."""
    return base64.b64encode(vec.astype(np.float32).tobytes()).decode("ascii")


def _decode_embedding(b64: str) -> np.ndarray:
    """Deserialize a base64 string back to a float32 numpy vector."""
    raw = base64.b64decode(b64)
    return np.frombuffer(raw, dtype=np.float32).copy()
