"""
CLIP text (and image) encoder for the retriever.

This is a thin wrapper around the same openai/CLIP library used in the indexer.
The vectors produced here are directly comparable to those stored in ChromaDB
because they come from the same model weights — the comparison is only valid
if the same model name is used in both the indexer and the retriever, which is
enforced by storing clip_model in the collection metadata.
"""

from __future__ import annotations

import torch
import numpy as np
from PIL import Image

import clip  # type: ignore


_model = None
_preprocess = None
_device: str = "cpu"
_loaded_model_name: str = ""


def load_model(model_name: str = "ViT-L/14", device: str = "cpu") -> None:
    global _model, _preprocess, _device, _loaded_model_name

    if _model is not None and _loaded_model_name == model_name:
        return

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    _device = device
    _model, _preprocess = clip.load(model_name, device=device)
    _model.eval()
    _loaded_model_name = model_name


def encode_text(text: str) -> np.ndarray:
    """Return an L2-normalized float32 embedding for a text string."""
    if _model is None:
        raise RuntimeError("Call load_model() before encode_text().")

    tokens = clip.tokenize([text], truncate=True).to(_device)
    with torch.no_grad():
        embedding = _model.encode_text(tokens)

    vec = embedding.squeeze(0).float().cpu().numpy()
    return _l2_normalize(vec)


def encode_image(image: Image.Image) -> np.ndarray:
    """Return an L2-normalized float32 embedding for a PIL image."""
    if _model is None:
        raise RuntimeError("Call load_model() before encode_image().")

    tensor = _preprocess(image).unsqueeze(0).to(_device)
    with torch.no_grad():
        embedding = _model.encode_image(tensor)

    vec = embedding.squeeze(0).float().cpu().numpy()
    return _l2_normalize(vec)


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-10:
        return vec
    return vec / norm
