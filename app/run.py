"""
Start the Fashion Search web app.

Usage (from repo root):
    python app/run.py

Or with custom settings:
    python app/run.py --port 8080 --db-path ./chroma_db --clip-model ViT-B/32
"""

import argparse
import os
import sys
from pathlib import Path

# Add retriever/ to sys.path so we can import RetrieverConfig for the default.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "retriever"))


def main():
    # Import here so the path insert above takes effect first.
    from config import RetrieverConfig  # retriever/config.py

    parser = argparse.ArgumentParser(description="Start the Fashion Search API + UI")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--db-path", type=str, default="./chroma_db")
    parser.add_argument("--image-dir", type=str, default="./datasets/val_test2020/test")
    # Default comes from RetrieverConfig — the single source of truth.
    # Do NOT hardcode a model string here; change RetrieverConfig.clip_model instead.
    parser.add_argument("--clip-model", type=str, default=RetrieverConfig.clip_model)
    args = parser.parse_args()

    # Pass config to the FastAPI app via environment variables.
    os.environ["CHROMA_DB_PATH"] = args.db_path
    os.environ["IMAGE_DIR"] = args.image_dir
    os.environ["CLIP_MODEL"] = args.clip_model

    import uvicorn
    print(f"\n Fashion Search starting at http://{args.host}:{args.port}\n")
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=False,
        app_dir=str(_REPO_ROOT),
    )


if __name__ == "__main__":
    main()
