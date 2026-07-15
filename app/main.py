"""
FastAPI backend for the fashion image search UI.

Wraps the existing retriever pipeline into a REST API:
  GET  /              → serves the frontend HTML
  POST /search        → runs the two-stage retrieval, returns JSON results
  GET  /image/{name}  → serves a single image from the dataset directory

The retriever modules (CLIP, query parser, reranker) are loaded once at
startup and reused across requests — no per-request model loading.

Usage:
    cd app
    uvicorn main:app --reload --port 8000

Then open http://localhost:8000 in your browser.
"""

from __future__ import annotations

import base64
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import chromadb
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Path setup — add retriever/ to sys.path so we can import its modules.
# ---------------------------------------------------------------------------
_APP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _APP_DIR.parent
_RETRIEVER_DIR = _REPO_ROOT / "retriever"
sys.path.insert(0, str(_RETRIEVER_DIR))

from config import RetrieverConfig          # retriever/config.py
from embedder import load_model as load_clip, encode_text   # retriever/embedder.py
from query_parser import parse_query        # retriever/query_parser.py
from main import run_query                  # retriever/main.py


# ---------------------------------------------------------------------------
# Global state — loaded once at startup, shared across requests.
# ---------------------------------------------------------------------------
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and open the Chroma collection at server startup."""
    import torch

    db_path = os.environ.get("CHROMA_DB_PATH", str(_REPO_ROOT / "chroma_db"))
    clip_model = os.environ.get("CLIP_MODEL", "ViT-B/32")
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    print(f"[startup] Loading CLIP {clip_model} on {device} ...")
    load_clip(clip_model, device)

    print(f"[startup] Opening ChromaDB at {db_path} ...")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection("fashion_images")
    print(f"[startup] Collection has {collection.count()} documents.")

    _state["collection"] = collection
    _state["config"] = RetrieverConfig()
    _state["image_dir"] = Path(
        os.environ.get(
            "IMAGE_DIR",
            str(_REPO_ROOT / "datasets" / "val_test2020" / "test"),
        )
    )

    yield  # application runs

    print("[shutdown] Cleaning up.")
    _state.clear()


app = FastAPI(
    title="Fashion Image Search",
    description="Two-stage fashion retrieval: CLIP ANN + semantic reranking",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str
    k: int = 10


class AttributeMatch(BaseModel):
    query_label: str
    query_color: Optional[str]
    matched_region_label: str
    matched_region_color: Optional[str]
    garment_similarity: float
    color_distance: Optional[float]
    person_id: int


class SearchResult(BaseModel):
    rank: int
    image_id: str
    filename: str
    final_score: float
    stage1_score: float
    attribute_score: float
    setting_score: float
    matched_attributes: list[AttributeMatch]


class ParsedQueryInfo(BaseModel):
    garments: list[dict]
    setting: Optional[str]
    llm_succeeded: bool


class SearchResponse(BaseModel):
    query: str
    parsed: ParsedQueryInfo
    results: list[SearchResult]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the single-page frontend."""
    html_path = _APP_DIR / "static" / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/image/{filename}")
async def serve_image(filename: str):
    """Serve a dataset image by filename. Prevents path traversal."""
    # Sanitize — only allow the basename, no directory components.
    safe_name = Path(filename).name
    image_path = _state["image_dir"] / safe_name
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image {safe_name} not found.")
    return FileResponse(str(image_path), media_type="image/jpeg")


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    """
    Run the two-stage retrieval pipeline for a query.

    Stage 1: CLIP text embedding → HNSW ANN lookup (top-100 candidates).
    Stage 2: Semantic garment matching + RGB color distance reranking.

    Returns the top-k results with per-image score breakdowns.
    """
    if not _state:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    collection = _state["collection"]
    config = _state["config"]

    # Parse the query so we can return it to the frontend for display.
    groq_model = config.groq_model
    parsed = parse_query(req.query, groq_model=groq_model)

    # Override k if the user requested more than the default.
    config_copy = RetrieverConfig(
        top_k_final=req.k,
        top_k_stage1=max(100, req.k * 10),
    )

    ranked = run_query(req.query, config_copy, collection)

    results = []
    for rank, r in enumerate(ranked[: req.k], 1):
        filename = Path(r.image_path).name
        matched = [
            AttributeMatch(
                query_label=a.get("query_label", ""),
                query_color=a.get("query_color"),
                matched_region_label=a.get("matched_region_label", ""),
                matched_region_color=a.get("matched_region_color"),
                garment_similarity=round(float(a.get("garment_similarity", 0)), 3),
                color_distance=round(float(a["color_distance"]), 1)
                if a.get("color_distance") is not None
                else None,
                person_id=int(a.get("person_id", -1)),
            )
            for a in r.matched_attributes
        ]
        results.append(
            SearchResult(
                rank=rank,
                image_id=r.image_id,
                filename=filename,
                final_score=round(r.final_score, 4),
                stage1_score=round(r.stage1_score, 4),
                attribute_score=round(r.attribute_score, 4),
                setting_score=round(r.setting_score, 4),
                matched_attributes=matched,
            )
        )

    return SearchResponse(
        query=req.query,
        parsed=ParsedQueryInfo(
            garments=parsed.garments,
            setting=parsed.setting,
            llm_succeeded=parsed.llm_succeeded,
        ),
        results=results,
    )


@app.get("/health")
async def health():
    """Quick liveness check."""
    count = _state.get("collection", None)
    return {
        "status": "ok",
        "indexed_images": count.count() if count else 0,
    }
