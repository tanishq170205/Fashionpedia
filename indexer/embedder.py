"""
CLIP embedding module for the indexer.

Both full images and region crops are embedded with the same model so their
vectors live in the same space. This is what makes region-level CLIP cosine
similarity meaningful at retrieval time: a CLIP text embedding of "windbreaker"
can be compared directly against the CLIP visual embedding of a detected jacket
crop, because both were produced by the same ViT backbone.

We L2-normalize every output vector. ChromaDB is configured to use cosine
distance, which is equivalent to dot product on normalized vectors. Storing
normalized embeddings means we never have to normalize at query time and the
stored bytes are always ready for dot product.
"""

from __future__ import annotations

import torch
import numpy as np
from PIL import Image

# CLIP is installed from the OpenAI GitHub repo (see requirements.txt).
import clip  # type: ignore


_model = None
_preprocess = None
_device: str = "cpu"
_loaded_model_name: str = ""


def load_model(model_name: str = "ViT-L/14", device: str = "cuda") -> None:
    """
    Load the CLIP model into module-level state.

    Called once by main.py at startup. Subsequent calls are no-ops if the same
    model is already loaded, so it's safe to call from multiple modules.
    """
    global _model, _preprocess, _device, _loaded_model_name

    if _model is not None and _loaded_model_name == model_name:
        return

    # If CUDA was requested but is not available, fall back gracefully rather
    # than crashing — the indexer still works on CPU, just slower.
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    _device = device
    _model, _preprocess = clip.load(model_name, device=device)
    _model.eval()
    _loaded_model_name = model_name
    print(f"Loaded CLIP {model_name} on {device}.")


def encode_image(image: Image.Image) -> np.ndarray:
    """
    Return an L2-normalized float32 embedding for a PIL image.

    The image goes through CLIP's standard preprocessing (resize, center crop,
    normalize). For region crops the upstream caller is responsible for cropping
    before passing here — we do not modify the crop boundaries.
    """
    if _model is None:
        raise RuntimeError("Call load_model() before encode_image().")

    tensor = _preprocess(image).unsqueeze(0).to(_device)
    with torch.no_grad():
        embedding = _model.encode_image(tensor)

    vec = embedding.squeeze(0).float().cpu().numpy()
    return _l2_normalize(vec)


def encode_images_batch(images: list) -> list:
    """
    Encode a list of PIL images in a single CLIP forward pass.

    Batching all crops from one image together (full image + N garment crops)
    reduces N+1 separate forward passes to 1. On CPU this gives ~3-4x speedup
    for the CLIP step because matrix multiplications are far more efficient at
    batch size > 1 than repeated single-item calls.

    Returns a list of L2-normalized float32 numpy arrays, one per input image.
    """
    if _model is None:
        raise RuntimeError("Call load_model() before encode_images_batch().")
    if not images:
        return []

    tensors = torch.stack([_preprocess(img) for img in images]).to(_device)
    with torch.no_grad():
        embeddings = _model.encode_image(tensors)

    results = []
    for vec in embeddings:
        results.append(_l2_normalize(vec.float().cpu().numpy()))
    return results



def encode_text(text: str) -> np.ndarray:
    """
    Return an L2-normalized float32 embedding for a text string.

    Used during indexing only to encode region label text for storage. At
    retrieval time the retriever has its own embedder module for encoding query
    text, but it loads the same model weights, so the vectors are comparable.
    """
    if _model is None:
        raise RuntimeError("Call load_model() before encode_text().")

    tokens = clip.tokenize([text], truncate=True).to(_device)
    with torch.no_grad():
        embedding = _model.encode_text(tokens)

    vec = embedding.squeeze(0).float().cpu().numpy()
    return _l2_normalize(vec)


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-10:
        return vec
    return vec / norm
