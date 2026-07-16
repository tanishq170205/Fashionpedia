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
import json
import os
import re
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
    # Read the default from RetrieverConfig — the single source of truth.
    # Do NOT hardcode a model string here; change RetrieverConfig.clip_model instead.
    clip_model = os.environ.get("CLIP_MODEL", RetrieverConfig.clip_model)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[startup] Opening ChromaDB at {db_path} ...")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection("fashion_images")
    print(f"[startup] Collection has {collection.count()} documents.")

    # ---- Model mismatch guard (fix 1b) ----------------------------------------
    # The indexer stores clip_model in every document's metadata. Peek at the
    # first document and compare. A mismatch means query embeddings and index
    # embeddings live in different vector spaces — results would be garbage.
    if collection.count() > 0:
        peek = collection.peek(limit=1)
        if peek["metadatas"]:
            stored_model = peek["metadatas"][0].get("clip_model")
            if stored_model and stored_model != clip_model:
                raise RuntimeError(
                    f"[CLIP model mismatch] Collection was indexed with \"{stored_model}\" "
                    f"but \"{clip_model}\" was requested. "
                    f"Either re-index with --clip-model {clip_model} or "
                    f"pass CLIP_MODEL={stored_model} when starting the app."
                )
    # ---------------------------------------------------------------------------

    print(f"[startup] Loading CLIP {clip_model} on {device} ...")
    load_clip(clip_model, device)

    _state["collection"] = collection
    _state["config"] = RetrieverConfig(clip_model=clip_model, db_path=db_path)
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

    # Build a config that inherits all defaults but respects the user's k.
    base = _state["config"]
    config_copy = RetrieverConfig(
        clip_model=base.clip_model,
        db_path=base.db_path,
        groq_model=base.groq_model,
        top_k_final=req.k,
        top_k_stage1=max(100, req.k * 10),
        w_stage1=base.w_stage1,
        w_attribute=base.w_attribute,
        w_setting=base.w_setting,
        color_distance_threshold=base.color_distance_threshold,
        garment_similarity_threshold=base.garment_similarity_threshold,
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


# ---------------------------------------------------------------------------
# Evaluation endpoints — support the /eval.html web judgment UI.
# ---------------------------------------------------------------------------

class JudgmentRequest(BaseModel):
    query: str
    image_id: str
    relevant: bool


_JUDGMENTS_DIR = Path(__file__).resolve().parent.parent / "eval" / "judgments"


def _query_slug(query: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", query.lower()).strip("_")[:60]


@app.post("/eval/judge")
async def save_judgment(req: JudgmentRequest):
    """
    Save a single relevance judgment to eval/judgments/<slug>.json.
    Creates/updates the file atomically with a per-image entry.
    """
    _JUDGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = _query_slug(req.query)
    path = _JUDGMENTS_DIR / f"{slug}.json"

    data: dict = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}

    data[req.image_id] = req.relevant
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return {"saved": True, "slug": slug, "total_judged": len(data)}


@app.get("/eval/summary")
async def eval_summary():
    """
    Compute P@5 and P@10 for every query that has a judgment file.
    Returns per-query and overall means.
    """
    if not _JUDGMENTS_DIR.exists():
        return {"queries": [], "mean_p5": None, "mean_p10": None}

    BENCHMARK = [
        "a person in a bright yellow raincoat",
        "professional business attire inside a modern office",
        "someone wearing a blue shirt sitting on a park bench",
        "casual weekend outfit for a city walk",
        "a red tie and a white shirt in a formal setting",
        "a woman in a long yellow dress",
        "a model in a denim jacket and blue jeans",
        "a black blazer with white trousers on a runway",
        "someone wearing a red coat outdoors",
        "a floral dress at a fashion show",
    ]

    query_results = []
    p5_vals, p10_vals = [], []

    for query in BENCHMARK:
        slug = _query_slug(query)
        path = _JUDGMENTS_DIR / f"{slug}.json"
        if not path.exists():
            query_results.append({"query": query, "judged": 0, "p5": None, "p10": None})
            continue

        data = json.loads(path.read_text(encoding="utf-8"))
        # data is {image_id: bool} in insertion order (rank order).
        judgments = list(data.values())
        judged = len(judgments)

        p5  = sum(judgments[:5])  / 5  if judged >= 5  else None
        p10 = sum(judgments[:10]) / 10 if judged >= 10 else None

        if p5  is not None: p5_vals.append(p5)
        if p10 is not None: p10_vals.append(p10)

        query_results.append({"query": query, "judged": judged, "p5": p5, "p10": p10})

    return {
        "queries":   query_results,
        "mean_p5":   round(sum(p5_vals)  / len(p5_vals),  3) if p5_vals  else None,
        "mean_p10":  round(sum(p10_vals) / len(p10_vals), 3) if p10_vals else None,
    }


@app.get("/eval")
async def eval_page():
    """Serve the web-based evaluation UI."""
    from fastapi.responses import FileResponse
    eval_html = Path(__file__).resolve().parent / "static" / "eval.html"
    return FileResponse(str(eval_html))
