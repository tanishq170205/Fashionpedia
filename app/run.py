"""
Start the Fashion Search web app.

Usage (from repo root):
    python app/run.py

Or with custom settings:
    python app/run.py --port 8080 --db-path ./chroma_db
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Start the Fashion Search API + UI")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--db-path", type=str, default="./chroma_db")
    parser.add_argument("--image-dir", type=str, default="./datasets/val_test2020/test")
    parser.add_argument("--clip-model", type=str, default="ViT-B/32")
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
        app_dir=str(__import__('pathlib').Path(__file__).parent.parent),
    )

if __name__ == "__main__":
    main()
